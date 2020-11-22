from django.shortcuts import render

def main_view(request):
    return render(
        request,
        'pages/in_main.html'
    )