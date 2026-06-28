from flask import Blueprint, request, jsonify
from app.model import db, User, RoleEnum
from flask_jwt_extended import jwt_required, get_jwt

user_bp = Blueprint('user_bp', __name__, url_prefix='/users')

def admin_required():
    claims = get_jwt()
    if claims.get("role") != "admin":
        return False
    return True

@user_bp.route('/', methods=['GET'])
@jwt_required()
def get_users():
    if not admin_required():
        return jsonify({"status": 403, "message": "Akses ditolak. Hanya Admin."}), 403

    users = User.query.order_by(User.id.asc()).all()
    data = []
    for u in users:
        data.append({
            'id': u.id,
            'username': u.username,
            'email': u.email,
            'role': u.role.name if u.role else None,
            'ruangan': u.ruangan,  
            'created_at': "Tersedia" if u.id else None
        })

    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@user_bp.route('/<int:user_id>', methods=['GET'])
@jwt_required()
def get_user(user_id):
    # if not admin_required():
    #     return jsonify({"status": 403, "message": "Akses ditolak."}), 403

    user = User.query.get(user_id)
    if not user:
        return jsonify({"status": 404, "message": "User tidak ditemukan"}), 404

    data = {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role.name if user.role else None,
        'ruangan': user.ruangan
    }
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@user_bp.route('/', methods=['POST'])
@jwt_required()
def create_user():
    if not admin_required():
        return jsonify({"status": 403, "message": "Akses ditolak."}), 403

    data = request.get_json()
    if not data or not all(k in data for k in ['username', 'email', 'password']):
        return jsonify({'status': 400, 'message': 'Username, Email, Password wajib diisi'}), 400

    if User.query.filter_by(username=data['username']).first():
        return jsonify({'status': 409, 'message': 'Username sudah digunakan'}), 409

    if User.query.filter_by(email=data['email']).first():
        return jsonify({'status': 409, 'message': 'Email sudah digunakan'}), 409

    role_str = data.get('role', 'user')
    try:
        role_enum = RoleEnum[role_str]
    except KeyError:
        return jsonify({'status': 400, 'message': 'Role tidak valid (opsi: admin, user, ketim)'}), 400
    
    ruangan = data.get('ruangan', None)

    new_user = User(
        username=data['username'],
        email=data['email'],
        role=role_enum,
        ruangan=ruangan
    )
    new_user.set_password(data['password'])
    
    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        'status': 201, 
        'message': 'User berhasil dibuat', 
        'data': {'id': new_user.id, 'username': new_user.username, 'ruangan': new_user.ruangan}
    }), 201


# --- UPDATE (Edit User/Pindah Ruangan) ---
@user_bp.route('/<int:user_id>', methods=['PUT'])
@jwt_required()
def update_user(user_id):
    if not admin_required():
        return jsonify({"status": 403, "message": "Akses ditolak."}), 403

    user = User.query.get(user_id)
    if not user:
        return jsonify({"status": 404, "message": "User tidak ditemukan"}), 404

    data = request.get_json()
    if not data:
        return jsonify({'status': 400, 'message': 'Tidak ada data update'}), 400

    # Update Username
    if 'username' in data and data['username'] != user.username:
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'status': 409, 'message': 'Username sudah digunakan'}), 409
        user.username = data['username']

    # Update Email
    if 'email' in data and data['email'] != user.email:
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'status': 409, 'message': 'Email sudah digunakan'}), 409
        user.email = data['email']
        
    # Update Password
    if 'password' in data and data['password']:
        user.set_password(data['password'])
    
    # Update Role
    if 'role' in data:
        try:
            user.role = RoleEnum[data['role']]
        except KeyError:
            return jsonify({'status': 400, 'message': 'Role tidak valid'}), 400

    # REVISI: Update Ruangan (Fitur Pindah Tugas)
    if 'ruangan' in data:
        user.ruangan = data['ruangan']

    db.session.commit()
    return jsonify({'status': 200, 'message': 'User berhasil diperbarui', 'data': {'id': user.id, 'ruangan': user.ruangan}}), 200


# --- DELETE (Remove User) ---
@user_bp.route('/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    if not admin_required():
        return jsonify({"status": 403, "message": "Akses ditolak."}), 403

    user = User.query.get(user_id)
    if not user:
        return jsonify({"status": 404, "message": "User tidak ditemukan"}), 404

    db.session.delete(user)
    db.session.commit()
    return jsonify({'status': 200, 'message': 'User berhasil dihapus'}), 200