from flask_jwt_extended import verify_jwt_in_request, get_jwt
from functools import wraps
from flask import jsonify

def role_required(*required_roles):
    def wrapper(fn):
        @wraps(fn)
        def decorator(*args, **kwargs):
            verify_jwt_in_request()
            claims = get_jwt()  

            if claims.get("role") not in required_roles:
                return jsonify({"error": "Forbidden, role not allowed"}), 403

            return fn(*args, **kwargs)
        return decorator
    return wrapper
