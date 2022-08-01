from django.contrib.auth.forms import AuthenticationForm

class LoginForm(AuthenticationForm):
    """ ログインフォーム """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 全てのフォーム部品へ適用
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'
            # field.widget.attrs['placeholder'] = field.label
