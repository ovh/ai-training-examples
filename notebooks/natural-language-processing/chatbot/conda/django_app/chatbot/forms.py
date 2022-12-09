from django import forms


class ContactForm(forms.Form):
    user = forms.CharField()
    message = forms.CharField(widget=forms.Textarea)
