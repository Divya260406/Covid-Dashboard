from django.shortcuts import render, redirect
from django import forms
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
import json
import folium
import numpy as np
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score


class UploadFileForm(forms.Form):
    dataset = forms.FileField(
        label="Upload CSV Dataset",
        widget=forms.ClearableFileInput(attrs={'accept': '.csv'})
    )
    algorithm = forms.ChoiceField(
        choices=[('regression', 'Linear Regression'), ('logistic', 'Logistic Regression')],
        label="Select Algorithm"
    )


def load_covid_data():
    try:
        death_df = pd.read_excel('chartapp/data/deaths_global.xlsx')
        confirmed_df = pd.read_excel('chartapp/data/time_series_covid19_confirmed_global.xlsx')
        country_df = pd.read_excel('chartapp/data/cases_country.xlsx')
        
        # Data cleaning - normalize column names to lowercase and fix underscores
        mapping = {
            'Country_Region': 'country',
            'Last_Update': 'last_update',
            'Lat': 'lat',
            'Long_': 'long',
            'Confirmed': 'confirmed',
            'Deaths': 'deaths',
            'Active': 'active'
        }
        
        country_df = country_df.rename(columns=mapping)
        confirmed_df = confirmed_df.rename(columns={'Province/State': 'state', 'Country/Region': 'country'})
        death_df = death_df.rename(columns={'Province/State': 'state', 'Country/Region': 'country'})
        
        # Convert column names to lowercase
        country_df.columns = country_df.columns.str.lower()
        confirmed_df.columns = confirmed_df.columns.str.lower()
        death_df.columns = death_df.columns.str.lower()
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['confirmed', 'deaths', 'active']
        for col in numeric_columns:
            if col in country_df.columns:
                country_df[col] = pd.to_numeric(country_df[col], errors='coerce').fillna(0)
        
        return death_df, confirmed_df, country_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None


def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')
        
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'chartapp/login.html')

def register_view(request):
    if request.user.is_authenticated:
        return redirect('index')
        
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful!')
            return redirect('index')
        else:
            for error in form.errors.values():
                messages.error(request, error)
    
    return render(request, 'chartapp/register.html')

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def index(request):
    result = None
    error = None
    chart_url = None

    death_df, confirmed_df, country_df = load_covid_data()
    
    if country_df is None:
        return render(request, 'chartapp/index.html', {'error': 'Error loading COVID-19 data'})
    
    # Calculate totals and prepare data
    confirmed_total = int(country_df['confirmed'].sum())
    deaths_total = int(country_df['deaths'].sum())
    active_total = int(country_df['active'].sum())
    
    # Get top 10 countries
    sorted_country_df = country_df.sort_values('confirmed', ascending=False)
    top_10_countries = sorted_country_df.head(10)
    
    # Calculate daily new cases (global)
    confirmed_dates = confirmed_df.iloc[:, 4:].sum()
    daily_new_cases = confirmed_dates.diff().fillna(0)
    
    # Create daily new cases chart with dark theme
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=confirmed_df.columns[4:],
        y=daily_new_cases,
        name='Daily New Cases',
        fill='tonexty',
        line=dict(color='rgba(100,201,255,0.6)')
    ))
    fig_daily.update_layout(
        title='Global Daily New Cases',
        yaxis_title='Number of Cases',
        xaxis_title='Date',
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        )
    )
    daily_cases_chart = fig_daily.to_html()
    
    # Calculate and create CFR trend with dark theme
    death_dates = death_df.iloc[:, 4:].sum()
    cfr = (death_dates / confirmed_dates * 100).fillna(0)
    
    fig_cfr = go.Figure()
    fig_cfr.add_trace(go.Scatter(
        x=confirmed_df.columns[4:],
        y=cfr,
        name='Case Fatality Rate',
        line=dict(color='#ff6b6b')
    ))
    fig_cfr.update_layout(
        title='Case Fatality Rate (%) Over Time',
        yaxis_title='Percentage',
        xaxis_title='Date',
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        )
    )
    cfr_chart = fig_cfr.to_html()
    
    # Create confirmed cases chart with dark theme
    fig_confirmed = px.bar(
        top_10_countries,
        x="country",
        y="confirmed",
        title="Top 10 Countries - Confirmed Cases",
        color_discrete_sequence=["#64c9ff"]
    )
    fig_confirmed.update_layout(
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        )
    )
    confirmed_chart = fig_confirmed.to_html()
    
    # Create deaths chart with dark theme
    fig_deaths = px.bar(
        top_10_countries,
        x="country",
        y="deaths",
        title="Top 10 Countries - Deaths",
        color_discrete_sequence=["#ff6b6b"]
    )
    fig_deaths.update_layout(
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        )
    )
    deaths_chart = fig_deaths.to_html()
    
    # Create bubble chart with dark theme
    fig_bubble = px.scatter(
        top_10_countries,
        x="country",
        y="confirmed",
        size="confirmed",
        color="country",
        hover_name="country",
        title="COVID-19 Impact by Country",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_bubble.update_layout(
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        )
    )
    bubble_chart = fig_bubble.to_html()
    
    # Create world map with dark theme
    m = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles='CartoDB dark_matter'
    )
    for idx, row in confirmed_df.iterrows():
        try:
            if pd.notna(row['lat']) and pd.notna(row['long']):
                folium.Circle(
                    location=[row['lat'], row['long']],
                    radius=float(row.iloc[-1])/100,
                    color='#ff6b6b',
                    fill=True,
                    popup=f"{row['country']}<br>Confirmed: {row.iloc[-1]}"
                ).add_to(m)
        except Exception as e:
            continue
    
    map_html = m._repr_html_()
    
    # Prepare context for template
    context = {
        'confirmed_total': confirmed_total,
        'deaths_total': deaths_total,
        'active_total': active_total,
        'daily_cases_chart': daily_cases_chart,
        'cfr_chart': cfr_chart,
        'confirmed_chart': confirmed_chart,
        'deaths_chart': deaths_chart,
        'bubble_chart': bubble_chart,
        'world_map': map_html,
        'countries': sorted_country_df['country'].tolist()
    }
    
    return render(request, 'chartapp/index.html', context)

def get_country_data(request):
    if request.method == 'GET':
        country = request.GET.get('country', 'World')
        death_df, confirmed_df, country_df = load_covid_data()
        
        if confirmed_df is None:
            return JsonResponse({'error': 'Data not available'})
            
        try:
            if country == 'World':
                confirmed_data = confirmed_df.iloc[:, 4:].sum()
                death_data = death_df.iloc[:, 4:].sum()
            else:
                confirmed_data = confirmed_df[confirmed_df['country'] == country].iloc[:, 4:].sum()
                death_data = death_df[death_df['country'] == country].iloc[:, 4:].sum()
            
            dates = confirmed_df.columns[4:].tolist()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=confirmed_data, name='Confirmed', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dates, y=death_data, name='Deaths', line=dict(color='red')))
            
            fig.update_layout(
                title=f"COVID-19    Cases in {country}",
                xaxis_title='Date',
                yaxis_title='Number of Cases'
            )
            
            return JsonResponse({
                'chart': fig.to_html()
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request'})


def index_old(request):
    result = None
    error = None
    chart_url = None

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            algorithm = form.cleaned_data['algorithm']
            dataset_file = request.FILES['dataset']

            try:
                data = pd.read_csv(dataset_file)

                if data.shape[1] < 2:
                    error = "Dataset must have at least 2 columns: features and target."
                elif not all(data.dtypes.apply(pd.api.types.is_numeric_dtype)):
                    error = "Dataset must contain only numeric values."
                else:
                    data = data.dropna()
                    X = data.iloc[:, :-1]
                    y = data.iloc[:, -1]

                    if algorithm == 'logistic':
                        if y.nunique() > 2:
                            error = "Logistic Regression currently supports only binary classification."
                        y = y.astype(int)

                    if not error:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        if algorithm == 'regression':
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            score = r2_score(y_test, y_pred)
                            result = f'R2 Score (Regression): {score:.3f}'

                            # Line chart with grid for Linear Regression
                            plt.figure()
                            plt.plot(range(len(y_test)), y_test.values, label='Actual', marker='o')
                            plt.plot(range(len(y_pred)), y_pred, label='Predicted', marker='x')
                            plt.xlabel("Sample Index")
                            plt.ylabel("Target Value")
                            plt.title("Linear Regression: Actual vs Predicted")
                            plt.legend()
                            plt.grid(True)

                        else:
                            model = LogisticRegression(max_iter=1000)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            score = accuracy_score(y_test, y_pred)
                            result = f'Accuracy (Logistic Regression): {score:.3%}'

                            # Line chart with grid for Logistic Regression
                            plt.figure()
                            plt.plot(range(len(y_test)), y_test.values, label='Actual', marker='o')
                            plt.plot(range(len(y_pred)), y_pred, label='Predicted', marker='x', linestyle='--')
                            plt.xlabel("Sample Index")
                            plt.ylabel("Class")
                            plt.title("Logistic Regression: Actual vs Predicted")
                            plt.legend()
                            plt.grid(True)

                        # Convert chart to base64
                        buf = io.BytesIO()
                        plt.tight_layout()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        chart_url = base64.b64encode(buf.read()).decode('utf-8')
                        buf.close()
                        plt.close()

            except Exception as e:
                error = f"Error processing dataset: {str(e)}"
        else:
            error = "Invalid form submission."
    else:
        form = UploadFileForm()

    return render(request, 'chartapp/index.html', {
        'form': form,
        'result': result,
        'error': error,
        'chart_url': chart_url
    })
