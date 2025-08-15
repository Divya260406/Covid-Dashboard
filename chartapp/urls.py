from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('get_country_data/', views.get_country_data, name='get_country_data'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
