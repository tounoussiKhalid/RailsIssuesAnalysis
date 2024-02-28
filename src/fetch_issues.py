import requests
import json

def fetch_issues(repo='rails/rails', issues_count=500):
    issues = []
    page = 1
    while len(issues) < issues_count:
        url = f"https://api.github.com/repos/{repo}/issues?state=all&page={page}&per_page=100"
        response = requests.get(url)
        page_issues = response.json()
        if not page_issues or response.status_code != 200:
            break  # Break the loop if no more issues are returned or an error occurs
        issues.extend(page_issues)
        page += 1
        if len(page_issues) < 100:  # Break if last page
            break
    return issues[:issues_count]

def save_issues(issues, filename='data/issues.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(issues, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    #issues = fetch_issues()
    #save_issues(issues)
    #print(f"Fetched {len(issues)} issues.")
    # Replace 'user' and 'repo' with the actual user and repo names for Ruby on Rails
    releases = get_github_releases('rails', 'rails')

    for release in releases:
        print(release['tag_name'], release['published_at'])