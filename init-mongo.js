db = db.getSiblingDB("admin");
db.createUser({
    user: "root",
    pwd: "example",
    roles: [{ role: "root", db: "admin" }]
});

db = db.getSiblingDB("recommendation_system");
db.createUser({
    user: "root",
    pwd: "example",
    roles: [{ role: "readWrite", db: "recommendation_system" }]
});
