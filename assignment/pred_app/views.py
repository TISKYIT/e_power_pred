from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView, LogoutView
from django.views.generic import TemplateView

from . import forms
import os
from . import inference


class TopView(TemplateView):
    """ トップページ """
    template_name = 'pred_app/top.html'


class HomeView(LoginRequiredMixin, TemplateView):
    """ ホームページ """
    template_name = 'pred_app/home.html'


class ResultView(TemplateView):
    """ 推論結果ページ """
    template_name = 'pred_app/result.html'
    
    # CSVファイルをダウンロード
    inference.get_csv()

    # 推論結果をコンテキストへ追加
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["result"] = inference.get_pred()
        return context

class LoginView(LoginView):
    """ ログインページ """
    form_class = forms.LoginForm 
    template_name = 'pred_app/login.html'


class LogoutView(LoginRequiredMixin, LogoutView):
    """ ログアウトページ """
    template_name = 'pred_app/login.html'