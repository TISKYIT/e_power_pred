from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import User


class LoginForm(AuthenticationForm):
    """ ログインフォーム """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 全てのフォーム部品へ適用
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'
            # field.widget.attrs['placeholder'] = field.label


class SignUpForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password']