from rest_framework import permissions

from organizations.models import Organization


class IsAdminOrSubmittedBy(permissions.BasePermission):
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS and request.user.is_authenticated:
            return True

        if request.method == "POST" and request.user.is_authenticated:
            return True

        if request.user and request.user.is_staff:
            return True

        return False

    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS and request.user.is_authenticated:
            return True

        if request.user and request.user.is_staff:
            return True

        if request.user == obj.submitted_by and obj.status == "PENDING":
            return True

        return False


class IsAdminOrOrgAdmin(permissions.BasePermission):
    def has_permission(self, request, view):
        # Allow read permissions to all authenticated users for safe methods
        if request.method in permissions.SAFE_METHODS and request.user.is_authenticated:
            return True

        # Allow all permissions to staff users
        if request.user and request.user.is_staff:
            return True

        # For write operations, we'll check object-level permissions
        return True

    def has_object_permission(self, request, view, obj):
        # Allow read permissions to all authenticated users for safe methods
        if request.method in permissions.SAFE_METHODS and request.user.is_authenticated:
            return True

        # Allow all permissions to staff users
        if request.user and request.user.is_staff:
            return True

        # Check if the user is the admin of the organization
        if hasattr(obj, "admin"):
            return obj.admin == request.user

        # For organization membership checks
        if hasattr(obj, "organization") and hasattr(obj.organization, "admin"):
            return obj.organization.admin == request.user

        return False


class IsOrgAdmin(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        organization = view.kwargs.get("organization_id")
        return Organization.objects.get(id=organization).admin == request.user
