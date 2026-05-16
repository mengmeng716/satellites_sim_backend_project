"""
ASGI config for satellites_sim_backend project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/asgi/
"""

"""
WSGI config for satellites_sim_backend project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://channels.readthedocs.io/en/latest/deploying.html
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'satellites_sim_backend.settings')

# Ensure Django forms its native ASGI app first
django_asgi_app = get_asgi_application()

import simulation_api.routing

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            simulation_api.routing.websocket_urlpatterns
        )
    ),
})
