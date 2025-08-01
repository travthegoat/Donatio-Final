from rest_framework_nested import routers

from .views import EventListViewSet

router = routers.DefaultRouter()
router.register(r"events", EventListViewSet, basename="events")

urlpatterns = router.urls
