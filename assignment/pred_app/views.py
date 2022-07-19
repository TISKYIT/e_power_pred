from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView, LogoutView
from django.views.generic import TemplateView

from . import forms
import os
from . import demand_pred


class TopView(TemplateView):
    """ トップページ """
    template_name = 'pred_app/top.html'


class HomeView(LoginRequiredMixin, TemplateView):
    """ ホームページ """
    template_name = 'pred_app/home.html'


class ResultView(TemplateView):
    """ 推論結果ページ """
    template_name = 'pred_app/result.html'

    def get_context_data(self, **kwargs):
        # 学習実行
        trainX, trainY, testX, testY, scaler = demand_pred.create_dataset(look_back=3)
        demand_pred.train(trainX, trainY, look_back=3)

        context = super().get_context_data(**kwargs)
        context["result"] = '0000'

        return context


class LoginView(LoginView):
    """ ログインページ """
    form_class = forms.LoginForm 
    template_name = 'pred_app/login.html'


class LogoutView(LoginRequiredMixin, LogoutView):
    """ ログアウトページ """
    template_name = 'pred_app/login.html'