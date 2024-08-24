from django import forms

class UserInputForm(forms.Form):
    Day_Mins = forms.FloatField(label='Day Minutes')
    Day_Calls = forms.IntegerField(label='Day Calls')
    Day_Charge = forms.FloatField(label='Day Charge')
    
    Eve_Mins = forms.FloatField(label='Evening Minutes')
    Eve_Calls = forms.IntegerField(label='Evening Calls')
    Eve_Charge = forms.FloatField(label='Evening Charge')
    
    Night_Mins = forms.FloatField(label='Night Minutes')
    Night_Calls = forms.IntegerField(label='Night Calls')
    Night_Charge = forms.FloatField(label='Night Charge')
    
    Intl_Mins = forms.FloatField(label='International Minutes')
    Intl_Calls = forms.IntegerField(label='International Calls')
    Intl_Charge = forms.FloatField(label='International Charge')
