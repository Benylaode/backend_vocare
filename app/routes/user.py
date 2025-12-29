from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from app.model import db, User, RoleEnum  
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from app.model import User
from app.utils import role_required

user_bp = Blueprint('user_bp', __name__, url_prefix='/users')

@user_bp.route('/', methods=['GET'])
@jwt_required()
@role_required("admin")
def get_users():
    users = User.query.all()
    return jsonify([{
        'id': u.id,
        'username': u.username,
        'email': u.email,
        'role': u.role.name,
        'intervensi': u.intervensi,
        'CPPT': u.CPPT,
        'laporan': u.laporan,
        'patients': u.patients
    } for u in users]), 200

@user_bp.route('/<int:user_id>', methods=['GET'])
@jwt_required()
@role_required("admin", "user")
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role.name
    }), 200

@user_bp.route('/', methods=['POST'])

def create_user():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing required fields'}), 400

    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 409

    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 409

    try:
        role = RoleEnum[data.get('role', RoleEnum.user.name)]
    except KeyError:
        return jsonify({'error': 'Invalid role provided'}), 400
    
    new_user = User(
        username=data['username'],
        email=data['email'],
        role=role
    )
    new_user.set_password(data['password'])
    
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User created', 'id': new_user.id}), 201

@user_bp.route('/<int:user_id>', methods=['PUT'])
@jwt_required()
@role_required("user", "admin")
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided for update'}), 400

    if 'username' in data and data['username'] != user.username:
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 409
        user.username = data['username']

    if 'email' in data and data['email'] != user.email:
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 409
        user.email = data['email']
        
    if 'password' in data:
        user.set_password(data['password'])
    
    if 'role' in data:
        try:
            user.role = RoleEnum[data['role']]
        except KeyError:
            return jsonify({'error': 'Invalid role provided'}), 400

    db.session.commit()
    return jsonify({'message': 'User updated'}), 200

@user_bp.route('/<int:user_id>', methods=['DELETE'])
@jwt_required()
@role_required("admin")
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'User deleted'}), 200

