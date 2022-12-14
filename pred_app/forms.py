from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import User
from django import forms


class LoginForm(AuthenticationForm):
    # ログインフォーム
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 全てのテキストボックスへform-controlを設定
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'


class SignUpForm(UserCreationForm):
    # サインアップフォーム
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 全てのテキストボックスへform-controlを設定 
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'

    # パスワード入力・非表示対応
    password = forms.CharField(widget=forms.PasswordInput(), label='パスワード')
    
    class Meta:
        # ユーザー認証
        model = User
        # フィールド指定
        fields = ['username', 'password']
        # フィールド名指定
        labels = {'username':'ユーザーID'}