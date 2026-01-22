def test_register_perawat(client):
    res = client.post('/auth/register', json={
        "username": "suster_baru",
        "email": "suster@test.com",
        "password": "123",
        "ruangan": "Melati" # Test input ruangan
    })
    assert res.status_code == 201
    assert "access_token" in res.json

def test_login_check_ruangan(client):
    # Register dulu
    client.post('/auth/register', json={
        "username": "suster_anggrek", "email": "a@test.com", "password": "123", "ruangan": "Anggrek"
    })
    
    # Login
    res = client.post('/auth/login', json={
        "username": "suster_anggrek", "password": "123"
    })
    
    assert res.status_code == 200
    assert res.json['ruangan'] == "Anggrek" # Pastikan ruangan kembali di response