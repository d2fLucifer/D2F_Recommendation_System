from spark_session import create_spark_session
import google.generativeai as genai
from typing import List, Dict, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitError(Exception):
    pass

class DescriptionGenerator:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.min_delay = 0.5  # Chờ tối thiểu 0.5 giây giữa các lần request
        self.last_request_time = 0
        self._lock = threading.Lock()  # Khoá dùng để đồng bộ hoá
        self._setup_gemini()

    def _setup_gemini(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _wait_for_rate_limit(self):
        """Đảm bảo khoảng cách giữa các lần request không nhỏ hơn self.min_delay."""
        with self._lock:  # Giúp chỉ một luồng thao tác vào cùng thời điểm
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
            self.last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError)
    )
    def _generate_single_description(self, name: str) -> Optional[str]:
        """Tạo mô tả cho một sản phẩm với logic retry và rate limit."""
        try:
            self._wait_for_rate_limit()
            prompt = f"Generate a concise product description (at least 200 words) for: {name}"
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except google.api_core.exceptions.ResourceExhausted:
            logger.warning(f"Rate limit hit for {name}, retrying...")
            raise RateLimitError("API quota exceeded")
        except Exception as e:
            logger.error(f"Failed to generate description for {name}: {str(e)}")
            raise

    def _process_batch(self, names: List[str], max_workers: int = 2) -> Dict[str, str]:
        """Xử lý một danh sách tên (batch) bằng luồng, giới hạn số luồng để tránh quá tải."""
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(self._generate_single_description, name): name
                for name in names
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result() or "Description generation failed"
                except Exception:
                    results[name] = "Description generation failed"
        return results

def generate_descriptions(
    spark_session,
    input_path: str,
    api_key: str,
    batch_size: int = 2,
    checkpoint_path: Optional[str] = None
) -> None:
    """
    Hàm tạo mô tả sản phẩm với logic giới hạn tần suất, retry, và ghi checkpoint.
    """
    generator = DescriptionGenerator(api_key)
    
    try:
        df = spark_session.read.csv(input_path, header=True, inferSchema=True)
    except Exception as e:
        logger.error(f"Failed to read input data: {str(e)}")
        raise
        
    names = [row.name for row in df.select("name").collect()]
    total_batches = (len(names) + batch_size - 1) // batch_size
    all_descriptions = {}
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(names))
        batch = names[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_num + 1}/{total_batches}")
        batch_results = generator._process_batch(batch)
        all_descriptions.update(batch_results)
        
        # Ghi checkpoint sau mỗi batch (tuỳ chọn)
        if checkpoint_path:
            checkpoint_df = spark_session.createDataFrame(
                [(k, v) for k, v in all_descriptions.items()],
                ["name", "description"]
            )
            checkpoint_df.write.mode("overwrite").parquet(f"{checkpoint_path}/batch_{batch_num}")
    
    # Ghép mô tả mới với dữ liệu gốc
    description_data = [(name, all_descriptions.get(name, "")) for name in names]
    description_df = spark_session.createDataFrame(description_data, ["name", "description"])
    result_df = df.join(description_df, "name", "left")
    
    logger.info("Description generation complete!")
    result_df.show(truncate=False)
    result_df.write.mode("append").parquet("s3a://recommendation/new_ds/")
    
    return result_df

if __name__ == "__main__":
    spark_session = create_spark_session("Extract Data")
    result = generate_descriptions(
        spark_session=spark_session,
        input_path="s3a://recommendation/raw/new_dataset.csv",
        api_key="YOUR_GOOGLE_GENERATIVE_API_KEY_HERE",
        batch_size=2,
        checkpoint_path="s3a://recommendation/checkpoints/descriptions"
    )
