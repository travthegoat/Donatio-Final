from django.urls import include, path
from rest_framework_nested import routers

from activities.views import ActivityViewSet
from events.views import EventViewSet

from .views import (
    OrganizationChatViewSet,
    OrganizationRequestViewSet,
    OrganizationViewSet,
)

router = routers.DefaultRouter()
router.register(
    r"organization-requests",
    OrganizationRequestViewSet,
    basename="organization-requests",
)
router.register(r"organizations", OrganizationViewSet, basename="organizations")

# Nested router for organization-specific resources
organization_router = routers.NestedDefaultRouter(
    router, r"organizations", lookup="organization"
)
organization_router.register(
    r"chats", OrganizationChatViewSet, basename="organization-chats"
)
organization_router.register(r"events", EventViewSet, basename="organization-events")
organization_router.register(
    r"activities", ActivityViewSet, basename="organization-activities"
)

# Main URL patterns
urlpatterns = [
    path("", include(router.urls)),
    path("", include(organization_router.urls)),
]
