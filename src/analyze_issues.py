import requests
from datetime import datetime, timedelta
import pandas as pd
import json

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


def get_github_releases(user, repo):
    releases = requests.get(f"https://api.github.com/repos/{user}/{repo}/releases")
    if releases.ok:
        return releases.json()
    else:
        print(f"Error: {releases.status_code}")
        return None
    
def analyze_issues_with_releases(user, repo, issues):
    # Get the release data
    releases = get_github_releases(user, repo)
    if releases is None:
        return "Error retrieving release data."

    # Parse release dates and sort them
    release_dates = {release['tag_name']: release['published_at'] for release in releases}
    release_dates = {k: v for k, v in sorted(release_dates.items(), key=lambda item: item[1])}
    
    # Create a DataFrame for analysis
    df_releases = pd.DataFrame(list(release_dates.items()), columns=['Version', 'ReleaseDate'])
    df_releases['ReleaseDate'] = pd.to_datetime(df_releases['ReleaseDate'])
    
    # Convert issues into a DataFrame
    df_issues = pd.DataFrame(issues)
    df_issues['CreatedAt'] = pd.to_datetime(df_issues['created_at'])
    
    # Analyze the number of issues reported after each release
    analysis_results = []
    for i, release in df_releases.iterrows():
        release_date = release['ReleaseDate']
        version = release['Version']
        # Calculate the number of issues reported within a window after the release
        window_start = release_date
        window_end = window_start + timedelta(days=30)  # 30-day window after release
        issues_count = df_issues[(df_issues['CreatedAt'] > window_start) & (df_issues['CreatedAt'] <= window_end)].shape[0]
        analysis_results.append((version, release_date, issues_count))
    
    # Convert analysis results to DataFrame for easy viewing
    df_analysis = pd.DataFrame(analysis_results, columns=['Version', 'ReleaseDate', 'IssuesCountAfterRelease'])
    
    return df_analysis


if __name__ == "__main__":
    user = 'rails'
    repo = 'rails'
    df = load_issues_data()
    analysis_results = analyze_issues_with_releases(user, repo, df)
    print(analysis_results)