from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from app.model import db, User, RoleEnum
from datetime import timedelta

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or "username" not in data or "email" not in data or "password" not in data:
        return jsonify({"error": "username, email, and password are required"}), 400

    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "Username already exists"}), 400

    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already exists"}), 400

    role = RoleEnum.user   
    # REVISI: Ambil ruangan dari request untuk user ini
    ruangan = data.get("ruangan", None)

    new_user = User(
        username=data["username"],
        email=data["email"],
        role=role,
        ruangan=ruangan 
    )
    new_user.set_password(data["password"])
    db.session.add(new_user)
    db.session.commit()

    # Sertakan info ruangan di token (opsional, tapi berguna)
    access_token = create_access_token(
        identity=str(new_user.id),
        additional_claims={"role": new_user.role.name, "ruangan": new_user.ruangan}
    )

    return jsonify({
        "message": "User registered successfully!",
        "access_token": access_token
    }), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or "username" not in data or "password" not in data:
        return jsonify({"error": "username and password required"}), 400

    user = User.query.filter_by(username=data["username"]).first()
    if not user or not user.check_password(data["password"]):
        return jsonify({"msg": "Bad username or password"}), 401

    access_token = create_access_token(
        identity=str(user.id),
        additional_claims={"role": user.role.name, "ruangan": user.ruangan},
        expires_delta=timedelta(hours=8) # Masa aktif token disesuaikan shift kerja
    )

    return jsonify({
        "access_token": access_token,
        "username": user.username,
        "email": user.email,
        "role": user.role.name,
        "id": user.id,
        "ruangan": user.ruangan # Frontend perlu ini untuk validasi
    }), 200


@auth_bp.route("/profile", methods=["GET"])
@jwt_required()
def profile():
    user_id = int(get_jwt_identity())   
    claims = get_jwt()                  
    role = claims["role"]

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": role,
        "ruangan": user.ruangan # Tampilkan ruangan di profile
    }), 200