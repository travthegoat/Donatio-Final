from django.shortcuts import get_object_or_404
from django.utils import timezone
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import viewsets
from rest_framework.filters import SearchFilter
from rest_framework.permissions import IsAuthenticated

from core.permissions import IsOrgAdmin
from events.constants import EventStatusChoices
from events.models import Event
from events.serializers import EventSerializer
from organizations.models import Organization


class EventViewSet(viewsets.ModelViewSet):
    queryset = Event.objects.all()
    serializer_class = EventSerializer
    permission_classes = [IsAuthenticated, IsOrgAdmin]
    filter_backends = [SearchFilter, DjangoFilterBackend]
    search_fields = ["title"]
    filterset_fields = ["status"]
    http_method_names = ["get", "patch", "post"]

    def get_permissions(self):
        if self.action == "list":
            return [IsAuthenticated()]
        return super().get_permissions()

    def get_queryset(self):
        if self.action == "list":
            return Event.objects.filter(organization=self.kwargs.get("organization_pk"))
        return super().get_queryset()

    def get_serializer_context(self):
        context = super().get_serializer_context()

        organization_pk = self.kwargs.get("organization_pk")
        self.organization = get_object_or_404(Organization, pk=organization_pk)
        context["organization"] = self.organization

        return context

    def perform_create(self, serializer):
        organization = self.organization
        serializer.save(
            organization=organization,
            start_date=timezone.now(),
        )


class EventListViewSet(viewsets.ModelViewSet):
    serializer_class = EventSerializer
    permission_classes = [IsAuthenticated]
    http_method_names = ["get"]
    filter_backends = [SearchFilter]
    search_fields = ["title"]

    def get_queryset(self):
        if self.action == "list":
            return Event.objects.filter(status=EventStatusChoices.OPEN)
        return Event.objects.all()
