from flask import Flask


def test_create_app_exposes_liveness(app: Flask) -> None:
    response = app.test_client().get("/health/live")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_create_app_has_no_cors_header_by_default(app: Flask) -> None:
    response = app.test_client().get(
        "/health/live",
        headers={"Origin": "https://other.test"},
    )

    assert "Access-Control-Allow-Origin" not in response.headers
