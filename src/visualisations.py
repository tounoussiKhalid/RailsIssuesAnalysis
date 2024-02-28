import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
import matplotlib.dates as mdates
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pymannkendall as mk
from numpy.fft import rfft, rfftfreq


def find_repeated_issues(df_issues):
    # Convert the list of issue dictionaries to a DataFrame
    #df_issues = pd.DataFrame(issue_list)
    
    # Check for duplicated 'id' values
    duplicated_ids = df_issues[df_issues.duplicated('title', keep=False)]
    
    if not duplicated_ids.empty:
        print("Repeated issue IDs found:")
        print(duplicated_ids[['id', 'title']])
    else:
        print("No repeated issue number found.")

def analyse_over_time(df_issues):
    # Convert 'created_at' and 'closed_at' to datetime
    df_issues['created_at'] = pd.to_datetime(df_issues['created_at'])
    df_issues['closed_at'] = pd.to_datetime(df_issues['closed_at'])
    
    # Calculate resolution time in days
    df_issues['resolution_time'] = (df_issues['closed_at'] - df_issues['created_at']).dt.days
    
    # Filter out issues that are still open
    resolved_issues = df_issues.dropna(subset=['closed_at'])
    
    # Ensure resolution time is positive
    resolved_issues = resolved_issues[resolved_issues['resolution_time'] >= 0]
    
    # Plotting
    plt.figure(figsize=(14, 6))
    
    # Histogram of resolution times
    plt.hist(resolved_issues['resolution_time'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Issue Resolution Times')
    plt.xlabel('Resolution Time (days)')
    plt.ylabel('Number of Issues')
    
    # Calculate and plot median resolution time
    median_resolution_time = resolved_issues['resolution_time'].median()
    plt.axvline(median_resolution_time, color='red', linestyle='dashed', linewidth=2)
    plt.text(median_resolution_time, plt.ylim()[1]*0.9, f'Median: {median_resolution_time} days', color='red')
    
    plt.tight_layout()
    plt.show()


def load_issues_data(filepath='data/issues.json'):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
    else:
        raise ValueError("The 'created_at' column is missing from the data")
    df.set_index('created_at', inplace=True, drop=False)
    return df

def get_average_time_to_close(df_issues):
    # Convert 'created_at' and 'closed_at' to datetime if they aren't already
    df_issues['created_at'] = pd.to_datetime(df_issues['created_at'])
    df_issues['closed_at'] = pd.to_datetime(df_issues['closed_at'])

    # Filter out issues that are still open and issues without a closed_at date
    closed_issues = df_issues.dropna(subset=['closed_at']).copy()  # Added .copy() to explicitly create a copy

    # Calculate the time to close for each issue
    closed_issues['time_to_close'] = closed_issues['closed_at'] - closed_issues['created_at']

     # Calculate the average time to close
    average_time_to_close = closed_issues['time_to_close'].mean()

    # Calculate the median time to close
    median_time_to_close = closed_issues['time_to_close'].median()

    # Calculate the 90th percentile time to close
    ninetieth_percentile_time_to_close = closed_issues['time_to_close'].quantile(0.9)

    # Convert timedeltas to a string format for printing if necessary
    # For example, convert to total seconds and then to days, hours, minutes
    avg_time_str = str(average_time_to_close)
    med_time_str = str(median_time_to_close)
    ninety_pct_time_str = str(ninetieth_percentile_time_to_close)

    # Print the average, median, and 90th percentile times to close with explanations
    print(f"Average time to close: {avg_time_str}")
    print(f"Median time to close: {med_time_str}")
    print("The median offers a better measure of a typical issue's resolution time, "
          "especially if you have a skewed distribution where you might have a few "
          "issues that took an exceptionally long time to close.")
    print(f"90th percentile time to close: {ninety_pct_time_str}")
    print("The 90th percentile helps in understanding the upper bounds of time taken for most "
          "issues (90%) and is particularly useful in service level agreements (SLAs) to ensure "
          "that a high percentage of issues are resolved within a target time frame.")


def plot_issues_over_time(df):
     # Convert the 'created_at' and 'closed_at' to datetime objects
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['closed_at'] = pd.to_datetime(df['closed_at'])

    # Set the time period for the plot (e.g., daily)
    created_df = df.set_index('created_at').resample('D').size()
    closed_df = df.set_index('closed_at').resample('D').size()

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(created_df.index, created_df, label='Issues Created', marker='o')
    plt.plot(closed_df.index, closed_df, label='Issues Closed', marker='o')
    
    # Adding title and labels
    plt.title('Number of Issues Created and Closed Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Issues')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_issue_peaks(df):
    # Convert 'created_at' to datetime
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(15, 7))

    # Determine the bin range from the earliest to the latest date
    min_date = df['created_at'].min()
    max_date = pd.to_datetime("now").tz_localize(None)  # Use current time and normalize
    bins = pd.date_range(start=min_date, end=max_date, freq='D').to_pydatetime()

    # Plot histogram with the range from min_date to max_date
    df['created_at'].plot.hist(ax=ax, bins=bins, color='royalblue', alpha=0.7)

     # Set the x-axis formatter for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Optionally, set a locator for the x-axis to control the density of the labels
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Set labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of New Issues', fontsize=12)
    ax.set_title('Distribution of New Issues Reported Over Time', fontsize=14)

    # Improve grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Tight layout to ensure everything fits
    plt.tight_layout()

    # Save and show the figure
    plt.savefig('results/issues_daily_histogram.png', bbox_inches='tight')
    plt.show()

def plot_hourly_issue_peaks(df):
    # Convert 'created_at' to datetime and extract date and hour
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    df['hour'] = df['created_at'].dt.hour

    # Create a pivot table counting the number of issues for each hour and date
    heatmap_data = df.pivot_table(index='hour', columns='date', values='id', aggfunc='count', fill_value=0)

    # Plot the heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=.5, annot=True, fmt='d')

    # Set the titles and labels
    plt.title('Hourly Distribution of New Issues Reported Each Day', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Hour of Day', fontsize=12)

    # Improve the layout and appearance
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save and show the figure
    plt.savefig('results/issues_hourly_heatmap.png', bbox_inches='tight')
    plt.show()

def analysePicks(df):
    # Set the date as the index
    df.set_index('created_at', inplace=True)

    # Ensure the data is in a regular frequency, resampling if necessary
    df_resampled = df['id'].resample('D').count()  # 'id' should be a column representing unique issue IDs

    # Decompose the time series
    decomposition = seasonal_decompose(df_resampled, model='additive')

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

   # Plot the seasonal component
    plt.figure(figsize=(14,7))
    plt.plot(df_resampled, label='Original')
    plt.legend(loc='best')
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_top_issue_reporters(df):
    df['user_login'] = df['user'].apply(lambda x: x['login'])
    top_reporters = df['user_login'].value_counts().head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_reporters.values, y=top_reporters.index, palette='viridis')
    plt.title('Top 10 Issue Reporters')
    plt.xlabel('Number of Issues')
    plt.ylabel('User')
    plt.savefig('results/top_issue_reporters.png')
    plt.close()

def plot_popular_labels(df):
    # Extract 'name' from each label dictionary in the 'labels' column
    # and create a flattened list of all label names
    all_labels = pd.Series([label['name'] for sublist in df['labels'] for label in sublist])
    
    # Count occurrences of each label name to determine popularity
    label_counts = all_labels.value_counts().reset_index()
    label_counts.columns = ['label_name', 'count']
    
    # Plotting the treemap
    fig = px.treemap(label_counts, path=['label_name'], values='count',
                     title='Popularity of Issue Categories',
                     color='count', hover_data=['label_name'],
                      color_continuous_scale = 'RdBu')
    fig.write_image("results/popular_labels.png")

def time_series_check(df_issues):
    # Ensure 'created_at' is in datetime format
    df_issues['created_at'] = pd.to_datetime(df_issues['created_at'])
    
    # Create a series with 1's to count issues
    df_issues['count'] = 1
    
    # Resample to daily counts, excluding columns that can't be summed
    issues_per_time = df_issues.resample('D', on='created_at')['count'].sum()

    # Plot the time series
    issues_per_time.plot(figsize=(14, 7))
    plt.title('Daily Number of Issues Created')
    plt.xlabel('Date')
    plt.ylabel('Number of Issues')
    plt.show()

def seasonality_check(df_issues):
    issues_per_time = df_issues.resample('D', on='created_at')['count'].sum()
    # Ensure the index has a frequency set (daily in this case)
    issues_per_time = issues_per_time.asfreq('D')
    
    # Handle missing values if there are any
    issues_per_time.fillna(0, inplace=True)

    # Perform seasonal decomposition
    result = seasonal_decompose(issues_per_time, model='additive')

    # Plot the seasonal decomposition
    result.plot()
    plt.show()

def autocorrelation(df_issues):
    # Create a daily time series of issue counts
    daily_issue_counts = df_issues.set_index('created_at').resample('D').size()
    
    # Plot the daily issue counts to visualize the time series
    plt.figure(figsize=(12, 6))
    daily_issue_counts.plot(title='Daily Issue Counts')
    plt.xlabel('Date')
    plt.ylabel('Number of Issues')
    plt.show()

    # Plot Autocorrelation Function (ACF)
    plt.figure(figsize=(12, 6))
    plot_acf(daily_issue_counts, lags=30)  # You can adjust the number of lags as needed
    plt.title('Autocorrelation Function (ACF) for Daily Issue Counts')
    plt.show()

    # Plot Partial Autocorrelation Function (PACF)
    plt.figure(figsize=(12, 6))
    plot_pacf(daily_issue_counts, lags=23)  # You can adjust the number of lags as needed
    plt.title('Partial Autocorrelation Function (PACF) for Daily Issue Counts')
    plt.show()

def rate_range_detection():
    # Assuming 'count' is the column with the number of issues
    signal = issues_per_time['count'].values
    model = "l2"  # Least squares detection model
    algo = Pelt(model=model).fit(signal)
    result = algo.predict(pen=10)

    # Display change points
    print(f"Change points at: {result}")

def statistical_testing(df_issues):
    df_issues['created_at'] = pd.to_datetime(df_issues['created_at'])
    
    # Create a series with 1's to count issues
    df_issues['count'] = 1
    
    # Resample to daily counts
    issues_per_time = df_issues.resample('D', on='created_at')['count'].sum()

    # Perform the Mann-Kendall test on the daily issue counts
    result = mk.original_test(issues_per_time)
    
    print(result)

def statistical_testing_hourly(df_issues):
    # Ensure 'created_at' is in datetime format
    df_issues['created_at'] = pd.to_datetime(df_issues['created_at'])
    
    # Create a series with 1's to count issues
    df_issues['count'] = 1
    
    # Resample to hourly counts
    issues_per_time_hourly = df_issues.resample('h', on='created_at')['count'].sum()

    # Perform the Mann-Kendall test on the hourly issue counts
    result = mk.original_test(issues_per_time_hourly)
    
    print(result)

def statistical_testing_timely(df_issues):
    # Ensure 'created_at' is in datetime format
    df_issues['created_at'] = pd.to_datetime(df_issues['created_at'])
    
    # Create a series with 1's to count issues
    df_issues['count'] = 1
    
    df_issues['hour'] = df_issues['created_at'].dt.hour
    df_issues['time_block'] = pd.cut(df_issues['hour'], bins=[0,6,12,18,23], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)
    issues_by_time_block = df_issues.groupby('time_block')['count'].sum()

    # Perform the Mann-Kendall test on the hourly issue counts
    result = mk.original_test(issues_by_time_block)
    
    print(result)

def fourier_hourly_analysis(df_issues):
    df_issues['created_at'] = pd.to_datetime(df_issues['created_at'])

    # Adding a count column for each issue
    df_issues['count'] = 1

    # Resampling to get hourly counts
    issues_per_time_hourly = df_issues.resample('H', on='created_at')['count'].sum()

    # Assuming issues_per_time_hourly is a Series with hourly data
    y = issues_per_time_hourly.values  # Use .values to get the numpy array

    # Applying Fast Fourier Transform
    y_fft = rfft(y)

    # Calculating frequencies
    # The sampling spacing 'd' is 1 hour in this case. If you want it in minutes, use 1/60 as the example provided.
    freq = rfftfreq(len(y), d=1)  # Use d=1/60 if you prefer the frequency in cycles per minute

    # Magnitude of FFT components (for plotting)
    y_fft_mag = np.abs(y_fft)

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.stem(freq, y_fft_mag)  # Using stem plot to better visualize the discrete nature of the frequencies
    plt.title('FFT of Hourly Issues Count')
    plt.xlabel('Frequency (cycles per hour)')
    plt.ylabel('Magnitude')
    plt.show()

def fourier_daily_analysis(df_issues):
    df_issues['created_at'] = pd.to_datetime(df_issues['created_at'])

    # Adding a count column for each issue
    df_issues['count'] = 1

    # Resampling to get daily counts
    issues_per_time_daily = df_issues.resample('D', on='created_at')['count'].sum()

    # Assuming issues_per_time_daily is a Series with daily data
    y = issues_per_time_daily.values  # Use .values to get the numpy array

    # Applying Fast Fourier Transform
    y_fft = rfft(y)

    # Calculating frequencies
    # The sampling spacing 'd' is 1 day in this case.
    freq = rfftfreq(len(y), d=1)  # The frequency unit is cycles per day

    # Magnitude of FFT components (for plotting)
    y_fft_mag = np.abs(y_fft)

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.stem(freq, y_fft_mag)  # Using stem plot to better visualize the discrete nature of the frequencies
    plt.title('FFT of Daily Issues Count')
    plt.xlabel('Frequency (cycles per day)')
    plt.ylabel('Magnitude')
    plt.show()

def fourier_part_of_day_analysis(df_issues, start_hour, end_hour):
    # Convert to datetime and filter times
    df_issues['created_at'] = pd.to_datetime(df_issues['created_at'])
    df_issues['hour'] = df_issues['created_at'].dt.hour

    # Filter based on the part of the day
    part_of_day_issues = df_issues[(df_issues['hour'] >= start_hour) & (df_issues['hour'] < end_hour)]

    # Adding a count column for each issue
    part_of_day_issues['count'] = 1

    # Resampling to get daily counts within the custom part of the day
    issues_per_time_daily = part_of_day_issues.resample('D', on='created_at')['count'].sum()

    # Assuming issues_per_time_daily is a Series with daily data
    y = issues_per_time_daily.values  # Use .values to get the numpy array

    # Applying Fast Fourier Transform
    y_fft = rfft(y)

    # Calculating frequencies
    freq = rfftfreq(len(y), d=1)  # The frequency unit is cycles per day

    # Magnitude of FFT components (for plotting)
    y_fft_mag = np.abs(y_fft)

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.stem(freq, y_fft_mag)  # Using stem plot to better visualize the discrete nature of the frequencies
    plt.title(f'FFT of Issues Count from {start_hour}:00 to {end_hour}:00')
    plt.xlabel('Frequency (cycles per day)')
    plt.ylabel('Magnitude')
    plt.show()



if __name__ == '__main__':
    df = load_issues_data()
    get_average_time_to_close(df)
    time_series_check(df)
    seasonality_check(df)
    autocorrelation(df)
    statistical_testing(df)
    statistical_testing_timely(df)
    fourier_daily_analysis(df)
    fourier_part_of_day_analysis(df, 6, 12)  # for morning part of the day (6 AM to 12 PM)
    fourier_part_of_day_analysis(df, 18, 23)  # for evening part of the day (6 PM to 11 PM)
    find_repeated_issues(df)
    analysePicks(df)
    #plot_issues_over_time(df)
    #plot_hourly_issue_peaks(df)
    #plot_issue_peaks(df)
    #plot_top_issue_reporters(df)
    #plot_popular_labels(df)