def navigation(request):
    nav = [
        {"name": "Projects", "url": "/", "match": "/"},
        {"name": "Allowlist", "url": "/allowlist/", "match": "/allowlist/"},
        {"name": "Model registry", "url": "/registry/", "match": "/registry/"},
    ]
    path = request.path
    for item in nav:
        item["active"] = (
            path == item["match"]
            if item["match"] == "/"
            else path.startswith(item["match"])
        )
    return {"nav_items": nav}
