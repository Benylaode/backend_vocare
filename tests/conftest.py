import pytest
from app import create_app, db
from app.model import User, RoleEnum

@pytest.fixture
def app():
    # Definisikan config testing DI SINI
    test_config = {
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:", # Gunakan RAM
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "JWT_SECRET_KEY": "test-secret-key-123"
    }
    
    # Kirim ke create_app
    app = create_app(test_config=test_config)

    with app.app_context():
        db.create_all() # Membuat tabel di SQLite memory
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def perawat_mawar_token(client):
    # Setup data perawat
    nurse = User(username="nurse_mawar", email="mawar@test.com", role=RoleEnum.user, ruangan="Mawar")
    nurse.set_password("nurse123")
    
    db.session.add(nurse)
    db.session.commit()
    
    # Login
    res = client.post('/auth/login', json={
        "username": "nurse_mawar", "password": "nurse123"
    })
    return res.json['access_token']