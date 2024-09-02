#!/usr/bin/env python
# coding: utf-8

# # End-to-End Covid Data Analytics
# 
# ### This project showcases the use of SQL, Python to analyze Covid data (infections, deaths and vaccinations). It also shows how to use PANEL library from Holoviz to build a data visualization app. 
# 

# <span style="font-size: 30px; color: blue; font-style: italic; font-weight: bold;">Data Collection</span>

# In[3]:


import mysql.connector
from bokeh.models.formatters import NumeralTickFormatter
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import panel as pn
import numpy as np
import holoviews as hv
from panel.interact import interact
pn.extension('tabulator')
pn.extension(loading_spinner = 'petal', loading_color = '#00aa41')
import hvplot.pandas
formatter = NumeralTickFormatter(format='0.0,a')
import geopandas as gpd
import hvplot.pandas
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
import xarray as xr


# ## Below code shows how we can pull data from MYSQL. 
# ## For deployment to Huggingface, we have exported the combined dataframe and will be using the data going further.

# In[5]:


# conn = mysql.connector.connect(
# host ="localhost",
# user = "root",
# password = "root",
# database = "covid_dataproject")

# mycursor = conn.cursor()
# mycursor.execute("SELECT * FROM finalcovidview")
# myresult = mycursor.fetchall()
# df = pd.DataFrame(myresult, columns=[desc[0] for desc in mycursor.description])
# mycursor.close()
# conn.close()

# df.isnull().sum()

# df.fillna(0, inplace=True)
# # Method 1: Convert specific columns
# columns_to_convert = ['Total_Cases', 'Total_deaths', 'popfullyvaccinated']
# df[columns_to_convert] = df[columns_to_convert].astype(int)

# df["Year_Month"] = df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).apply(lambda x: x.zfill(2))

# df.head()



# In[6]:


# conn = mysql.connector.connect(
# host ="localhost",
# user = "root",
# password = "root",
# database = "covid_dataproject")

# mycursor = conn.cursor()
# mycursor.execute("SELECT * FROM finalcovidview3")
# myresult = mycursor.fetchall()
# df2 = pd.DataFrame(myresult, columns=[desc[0] for desc in mycursor.description])
# mycursor.close()
# conn.close()

# df2.isnull().sum()

# df2.fillna(0, inplace=True)
# # Method 1: Convert specific columns
# columns_to_convert = ['Total_Cases', 'Total_deaths', 'popfullyvaccinated']
# df2[columns_to_convert] = df2[columns_to_convert].astype(int)
# df2["Year_Month"] = df2["YEAR"].astype(str) + "-" + df2["MONTH"].astype(str).apply(lambda x: x.zfill(2))
# df2['populationmillion'] = df2['populationmillion'].astype(int)


# ## For the visualization, we will be showing world map and show countries based on critiria's selected
# ## For this, we need world map

# In[7]:


# Load the world map shapefile
world_map = gpd.read_file("./ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
world_map = world_map[["NAME","SOV_A3","CONTINENT","LABEL_X","LABEL_Y"]]
from holoviews.util.transform import lon_lat_to_easting_northing
world_map['x'], world_map['y'] = lon_lat_to_easting_northing(world_map.LABEL_X, world_map.LABEL_Y)
world_map[['LABEL_X', 'LABEL_Y', 'x', 'y']]
world_map.rename(columns = {'NAME':'location'}, inplace=True)
world_map2 = world_map[["location","x","y"]]
world_map3 = world_map[["location","LABEL_X","LABEL_Y"]]


# In[8]:


# conn = mysql.connector.connect(
# host ="localhost",
# user = "root",
# password = "root",
# database = "covid_dataproject")

# mycursor = conn.cursor()
# mycursor.execute("SELECT * FROM finalcovidview4")
# myresult = mycursor.fetchall()
# df3 = pd.DataFrame(myresult, columns=[desc[0] for desc in mycursor.description])
# mycursor.close()
# conn.close()

# df3.isnull().sum()
# df3.fillna(0, inplace=True)


# ## SQL -> SQL Queries -> Python pulled data -> Saved to the folder

# In[9]:


df = pd.read_excel("./df.xlsx")
df2 = pd.read_excel("./df2.xlsx")
df3 = pd.read_excel("./df3.xlsx")


# In[10]:


location = df['location'].unique().tolist()
YEAR = df['YEAR'].unique().tolist()
MONTH = df['MONTH'].unique().tolist()
Year_Month = df['Year_Month'].unique().tolist()
continent = df2['continent'].unique().tolist()
# YEAR2 = df2['YEAR'].unique().tolist()
# MONTH2 = df2['MONTH'].unique().tolist()
Year_Month2 = df2['Year_Month'].unique().tolist()


# ## Styling setup

# In[11]:


css_sample = {
    'background-color' : '#DAA520',
    'border': '2px solid black',
    'color':'black',
    'padding':'15px 20px',
    'text-align':'center',
    'text-decoration':'none',
    'font-size':'20px',
    'font-family':'tahoma',
    'margin':'10px 50px',
    'cursor':'move'
}


# <span style="font-size: 30px; color: blue; font-style: italic; font-weight: bold;">Data Visualization setup</span>

# ## Design Panel widgets

# In[12]:


SelectLocation = pn.widgets.Select(name = 'Country', options = location,min_width=75 ,max_width = 300, value = 'India', styles = css_sample )
SelectYear_Month = pn.widgets.MultiSelect(name = 'Year_Month', options = Year_Month,min_width=75 ,max_width = 300, value = Year_Month, styles = css_sample ) 
SelectContinent= pn.widgets.Select(name = 'Continent', options = continent,min_width=75 ,max_width = 300, value = 'Asia', styles = css_sample )
SelectYear_Month2 = pn.widgets.MultiSelect(name = 'Year_Month', options = Year_Month2,min_width=75 ,max_width = 300, value = Year_Month2 , styles = css_sample)


# ## Design Interactive objects

# In[13]:


idf = df.interactive()
idf2 = df2.interactive()


# In[14]:


pipeline1 = (
idf[(idf.location == SelectLocation)
    &
    (idf.Year_Month.isin(SelectYear_Month))
]
)


# In[15]:


pipeline2 = (
idf2[(idf2.continent == SelectContinent )
    &
    (idf2.Year_Month.isin(SelectYear_Month2))
]
)


# In[16]:


pipeline3 = (
idf[(idf.location == SelectLocation)
]
    .assign(Covid_Cases_Percent=lambda x: (x['Total_Cases'] / x['population']) * 100)
    .assign(Covid_Deaths_Percent=lambda x: (x['Total_deaths'] / x['population']) * 100)
    .assign(Covid_Vaccinated_Percent=lambda x: (x['popfullyvaccinated'] / x['population']) * 100)
)


# In[17]:


pipeline4 = (
idf2[(idf2.continent == SelectContinent)
]
    .assign(Covid_Cases_Percent=lambda x: (x['Total_Cases'] / (x['populationmillion']*1e6)) * 100)
    .assign(Covid_Deaths_Percent=lambda x: (x['Total_deaths'] / (x['populationmillion']*1e6)) * 100)
    .assign(Covid_Vaccinated_Percent=lambda x: (x['popfullyvaccinated'] / (x['populationmillion']*1e6)) * 100)
)


# In[18]:


Total_Cases = pipeline1['Total_Cases'].max()
Total_Casesindi = pn.indicators.Number(name = "Total_Cases", value = Total_Cases/1e3, format = '{value:.0f}K',
                                  title_size = '15pt',
                                  font_size = '50pt',
                                   colors=[(1000, 'green'), (10000, 'gold'), (1000000, 'red')], align='center' )

Total_deaths = pipeline1['Total_deaths'].max()
Total_deathsindi = pn.indicators.Number(name = "Total_deaths", value = Total_deaths/1e3, format = '{value:.0f}K',
                                  title_size = '15pt',
                                  font_size = '50pt',
                                   colors=[(100, 'green'), (1000, 'gold'), (10000, 'red')], align='center')

# def zerofunc(Total_Cases,Total_deaths):
#     if Total_Cases == 0:
#         return 0
#     else:
#         return (Total_deaths/Total_Cases)*100

Death_Percent = (Total_deaths/Total_Cases)*100
Death_Percentindi = pn.indicators.Number(name = "Death_Percent", value = Death_Percent, format = '{value:.1f}%',
                                  title_size = '15pt',
                                  font_size = '50pt',
                                   colors=[(0, 'green'), (2, 'gold'), (5, 'red')], align='center')


# In[19]:


Total_Cases2 = pipeline2['Total_Cases'].max()
Total_Casesindi2 = pn.indicators.Number(name = "Total_Cases", value = Total_Cases2/1e6, format = '{value:.0f}M',
                                  title_size = '15pt',
                                  font_size = '50pt',
                                   colors=[(10, 'green'), (100, 'gold'), (1000, 'red')], align='center')

Total_deaths2 = pipeline2['Total_deaths'].max()
Total_deathsindi2 = pn.indicators.Number(name = "Total_deaths", value = Total_deaths2/1e6, format = '{value:.0f}M',
                                  title_size = '15pt',
                                  font_size = '50pt',
                                   colors=[(0.5, 'green'), (1, 'gold'), (2, 'red')], align='center')

# def zerofunc(Total_Cases,Total_deaths):
#     if Total_Cases == 0:
#         return 0
#     else:
#         return (Total_deaths/Total_Cases)*100

Death_Percent2 = (Total_deaths2/Total_Cases2)*100
Death_Percentindi2 = pn.indicators.Number(name = "Death_Percent", value = Death_Percent2, format = '{value:.1f}%',
                                  title_size = '15pt',
                                  font_size = '50pt',
                                   colors=[(0, 'green'), (2, 'gold'), (5, 'red')], align='center')


# In[21]:


formatter = NumeralTickFormatter(format='0.0,a')
countrytrend1 = pipeline3.hvplot.line(y=['Total_Cases'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'green', fontscale = 1.4, yformatter=formatter, ylabel='Total_Cases',rot=90) 
countrytrend2 = pipeline3.hvplot.line(y=['Total_deaths'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'blue', fontscale = 1.4, yformatter=formatter, ylabel='Total_Deaths',rot=90)
countrytrend3 = pipeline3.hvplot.line(y=['popfullyvaccinated'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'brown', fontscale = 1.4, yformatter=formatter, ylabel='Total_Vaccinations',rot=90)
countrytrend4 = pipeline3.hvplot.line(y=['Covid_Cases_Percent'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'green', fontscale = 1.4, yformatter=formatter, ylabel='Cases%',rot=90) 
countrytrend5 = pipeline3.hvplot.line(y=['Covid_Deaths_Percent'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'blue', fontscale = 1.4, yformatter=formatter, ylabel='Death%',rot=90)
countrytrend6 = pipeline3.hvplot.line(y=['Covid_Vaccinated_Percent'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'brown', fontscale = 1.4, yformatter=formatter, ylabel='Vaccination%',rot=90)


# In[22]:


formatter = NumeralTickFormatter(format='0.0,a')
Continenttrend1 = pipeline4.hvplot.line(y=['Total_Cases'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'green', fontscale = 1.4, yformatter=formatter, ylabel='Total_Cases',rot=90) 
Continenttrend2 = pipeline4.hvplot.line(y=['Total_deaths'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'blue', fontscale = 1.4, yformatter=formatter, ylabel='Total_Deaths',rot=90)
Continenttrend3 = pipeline4.hvplot.line(y=['popfullyvaccinated'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'brown', fontscale = 1.4, yformatter=formatter, ylabel='Total_Vaccinations',rot=90)
Continenttrend4 = pipeline4.hvplot.line(y=['Covid_Cases_Percent'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'green', fontscale = 1.4, yformatter=formatter, ylabel='Cases%',rot=90) 
Continenttrend5 = pipeline4.hvplot.line(y=['Covid_Deaths_Percent'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'blue', fontscale = 1.4, yformatter=formatter, ylabel='Death%',rot=90)
Continenttrend6 = pipeline4.hvplot.line(y=['Covid_Vaccinated_Percent'],x='Year_Month',width=1600, xlabel="Month-Year", color = 'brown', fontscale = 1.4, yformatter=formatter, ylabel='Vaccination%',rot=90)


# In[24]:


min_valueInfected = df3['PercentInfected'].min()
max_valueInfected = df3['PercentInfected'].max()
min_valueDied = df3['PercentDied'].min()
max_valueDied = df3['PercentDied'].max()
min_valuevacc = df3['Percentvaccinated'].min()
max_valuevacc = df3['Percentvaccinated'].max()


# In[25]:


Infected_widget = pn.widgets.FloatInput(name='Minimum Infected Population', value= 10, step=0.1, start= 0, end=100)
Death_widget = pn.widgets.FloatInput(name='Minimum Died Population', value= 0.1, step=0.1, start= 0, end=100)
Vaccination_widget = pn.widgets.FloatInput(name='Minimum Vaccinated Population', value= 50, step=0.1, start= 0, end=100)


# In[26]:


dfworld = df3.merge(world_map2, on="location", how="left")
dfworld.head()


# In[27]:


idf3 = dfworld.interactive()


# In[28]:


pipeline5 = (
idf3[(idf3.location == SelectLocation)
])


# In[29]:


infectionbyloc = idf3[idf3.PercentInfected > Infected_widget]


# In[30]:


Infectionplot = infectionbyloc.hvplot.points('x', 'y', tiles=True, color='red', alpha=0.2, width=1400, height=800)


# In[31]:


Deathsbyloc = idf3[idf3.PercentDied > Death_widget]
Deathsbylocplot = Deathsbyloc.hvplot.points('x', 'y', tiles=True, color='red', alpha=0.2, width=1400, height=800)


# In[32]:


Vaccinationsbyloc = idf3[idf3.Percentvaccinated > Vaccination_widget]
Vaccinationsbylocplot = Vaccinationsbyloc.hvplot.points('x', 'y', tiles=True, color='red', alpha=0.2, width=1400, height=800)


# <span style="font-size: 30px; color: blue; font-style: italic; font-weight: bold;">Panel App Setup</span>

# In[33]:


pn.param.ParamMethod.loading_indicator = True

GoldenTemplate = pn.template.GoldenTemplate(title = "Covid Pandemic Data Dashboard", sidebar_width = 25, main_max_width = "2500px",
                                           header_background = '#DAA520')


# In[34]:


component1 = pn.Column(pn.Row(pn.Spacer(width=400),pn.pane.Markdown(" # Country Level Covid Statistics ")),
                       pn.Row(pn.Column(SelectLocation),pn.Spacer(width=400),pn.Column(SelectYear_Month)),
                       pn.Spacer(height=20),
                       pn.Row(pn.Spacer(width=50), pn.Column(Total_Casesindi),pn.Spacer(width=300),pn.Column(Total_deathsindi),pn.Spacer(width=300),pn.Column(Death_Percentindi)),
                       pn.Spacer(height=20),
                       pn.Row(pn.pane.Markdown(" --- ", width = 2500)),
                       pn.Row(pn.Spacer(width=400),pn.pane.Markdown("# Continent Level Covid Statistics")),
                       pn.Row(pn.Column(SelectContinent),pn.Spacer(width=400),pn.Column(SelectYear_Month2)),
                       pn.Spacer(height=20),
                       pn.Row(pn.Spacer(width=50), pn.Column(Total_Casesindi2),pn.Spacer(width=300),pn.Column(Total_deathsindi2),pn.Spacer(width=300),pn.Column(Death_Percentindi2)),
                       name = "Covid High Level Stats")


# In[35]:


component2 = pn.Column(pn.Row(pn.Spacer(width=400),pn.pane.Markdown(" # Country Level Covid Trend ")),
                       pn.Row(pn.Tabs(pn.Column(countrytrend1, name = "Total Cases Trend"),
                                     pn.Column(countrytrend2, name = "Total Deaths Trend"),
                                     pn.Column(countrytrend3, name = "Total Vaccinations Trend"))),
                       pn.Row(pn.Spacer(width=400),pn.pane.Markdown(" # Country Level Covid Percent Trend ")),
                       pn.Row(pn.Tabs(pn.Column(countrytrend4, name = "Covid Cases Percentage"),
                                     pn.Column(countrytrend5, name = "Covid Deaths Percentage"),
                                     pn.Column(countrytrend6, name = "Covid Vaccinations Percentage"))),
                       name = "Country Level Trends")


# In[36]:


component3 = pn.Column(pn.Row(pn.Spacer(width=400),pn.pane.Markdown(" # Continent Level Covid Trend ")),
                       pn.Row(pn.Tabs(pn.Column(Continenttrend1, name = "Total Cases Trend"),
                                     pn.Column(Continenttrend2, name = "Total Deaths Trend"),
                                     pn.Column(Continenttrend3, name = "Total Vaccinations Trend"))),
                       pn.Row(pn.Spacer(width=400),pn.pane.Markdown(" # Continent Level Covid Percent Trend ")),
                       pn.Row(pn.Tabs(pn.Column(Continenttrend4, name = "Covid Cases Percentage"),
                                     pn.Column(Continenttrend5, name = "Covid Deaths Percentage"),
                                     pn.Column(Continenttrend6, name = "Covid Vaccinations Percentage"))),
                       name = "Continent Level Trends")


# In[37]:


component4 = pn.Column(pn.Tabs(pn.Column(Infectionplot, name = "Countries > Infection Rate"),
                               pn.Column(Deathsbylocplot, name = "Countries > Death Rate"),
                               pn.Column(Vaccinationsbylocplot, name = "Countries > Vaccination Rate")),
                       name = "Covid stats on a Map")


# In[38]:


GoldenTemplate.main.append(component1)
GoldenTemplate.main.append(component2)
GoldenTemplate.main.append(component3)
GoldenTemplate.main.append(component4)
GoldenTemplate.show()

