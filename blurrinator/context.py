def navigation(request):
    if not getattr(request, "user", None) or not request.user.is_authenticated:
        return {
            "nav_items": [
                {"name": "About", "url": "/about/", "active": request.path.startswith("/about/")},
            ]
        }

    nav = [
        {"name": "Projects", "url": "/", "match": "/"},
        {"name": "Allowlist", "url": "/allowlist/", "match": "/allowlist/"},
        {"name": "Model registry", "url": "/registry/", "match": "/registry/"},
        {"name": "About", "url": "/about/", "match": "/about/"},
    ]
    nav.append({"name": "Users", "url": "/users/", "match": "/users/"})
    path = request.path
    for item in nav:
        item["active"] = (
            path == item["match"]
            if item["match"] == "/"
            else path.startswith(item["match"])
        )
    return {"nav_items": nav}
