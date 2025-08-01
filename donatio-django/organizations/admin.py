from django.contrib import admin

from .models import Organization, OrganizationRequest

admin.site.register(OrganizationRequest)
admin.site.register(Organization)
