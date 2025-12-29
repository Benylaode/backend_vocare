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
    def serialize_relationship(obj_list):
        # convert list of objects to list of dicts (fallback to string)
        result = []
        for obj in obj_list:
            if hasattr(obj, '__dict__'):
                # take only column attributes (avoid SQLAlchemy internal attrs)
                result.append({
                    k: v.isoformat() if hasattr(v, 'isoformat') else v
                    for k, v in obj.__dict__.items()
                    if not k.startswith('_')
                })
            else:
                result.append(str(obj))
        return result

    users = User.query.all()
    data = []
    for u in users:
        data.append({
            'id': u.id,
            'username': u.username,
            'email': u.email,
            'role': u.role.name if u.role else None,
            'intervensi': serialize_relationship(u.intervensi),
            'CPPT': serialize_relationship(u.CPPT),
            'laporan': serialize_relationship(u.laporan),
            'patients': serialize_relationship(u.patients)
        })

    return jsonify(data), 200


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

