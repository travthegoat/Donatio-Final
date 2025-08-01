from rest_framework import permissions


class IsOrgAdmin(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        return obj.organization.admin == request.user
