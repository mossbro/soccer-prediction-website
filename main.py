import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load and prepare match data
df = pd.read_csv('data/results.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= pd.Timestamp.now() - pd.DateOffset(years=10)].reset_index(drop=True)
df = df.sort_values('date').reset_index(drop=True)

# Load and prepare FIFA ranking data
ranking_df = pd.read_csv('data/fifa_mens_rank.csv')

# Convert team column to string to avoid dtype issues
ranking_df['team'] = ranking_df['team'].astype(str)

# Filter to recent years and select relevant columns
ranking_df = ranking_df[ranking_df['date'] >= 2015]
ranking_df = ranking_df[['date', 'team', 'total.points']]
ranking_df = ranking_df.rename(columns={
    'date': 'year',
    'team': 'team',
    'total.points': 'points'
})

# Convert 'year' column to datetime properly
ranking_df['year'] = pd.to_datetime(ranking_df['year'], format='%Y', errors='coerce')

# Create match result label
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'H'
    elif row['home_score'] == row['away_score']:
        return 'D'
    else:
        return 'A'
df['result'] = df.apply(get_result, axis=1)

# Recent form + goal difference features
def compute_recent_features(df, n=5):
    recent_stats = []
    team_history = {}

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        match_date = row['date']

        home_wins = home_draws = home_losses = 0
        away_wins = away_draws = away_losses = 0
        home_goal_diff = 0
        away_goal_diff = 0

        if home in team_history:
            past_home = [m for m in team_history[home] if m['date'] < match_date]
            last_n_home = sorted(past_home, key=lambda x: x['date'], reverse=True)[:n]
            for match in last_n_home:
                if match['result'] == 'W':
                    home_wins += 1
                elif match['result'] == 'D':
                    home_draws += 1
                else:
                    home_losses += 1
                home_goal_diff += match['goal_diff']

        if away in team_history:
            past_away = [m for m in team_history[away] if m['date'] < match_date]
            last_n_away = sorted(past_away, key=lambda x: x['date'], reverse=True)[:n]
            for match in last_n_away:
                if match['result'] == 'W':
                    away_wins += 1
                elif match['result'] == 'D':
                    away_draws += 1
                else:
                    away_losses += 1
                away_goal_diff += match['goal_diff']

        recent_stats.append({
            'recent_home_wins': home_wins,
            'recent_home_draws': home_draws,
            'recent_home_losses': home_losses,
            'recent_away_wins': away_wins,
            'recent_away_draws': away_draws,
            'recent_away_losses': away_losses,
            'recent_home_goal_diff': home_goal_diff,
            'recent_away_goal_diff': away_goal_diff,
        })

        # Determine match result for history update
        if row['home_score'] > row['away_score']:
            home_result = 'W'
            away_result = 'L'
        elif row['home_score'] == row['away_score']:
            home_result = 'D'
            away_result = 'D'
        else:
            home_result = 'L'
            away_result = 'W'

        home_goal_difference = row['home_score'] - row['away_score']
        away_goal_difference = row['away_score'] - row['home_score']

        # Update history
        team_history.setdefault(home, []).append({'date': match_date, 'result': home_result, 'goal_diff': home_goal_difference})
        team_history.setdefault(away, []).append({'date': match_date, 'result': away_result, 'goal_diff': away_goal_difference})

    recent_df = pd.DataFrame(recent_stats)
    return pd.concat([df.reset_index(drop=True), recent_df], axis=1)

df = compute_recent_features(df)

# Add FIFA ranking points with safe types
def get_ranking_points(team, match_date):
    team = str(team)  # ensure team is string
    match_date = pd.to_datetime(match_date)  # ensure date is datetime
    # Filter rows for this team and where ranking year <= match_date
    filtered = ranking_df[(ranking_df['team'] == team) & (ranking_df['year'] <= match_date)]
    if not filtered.empty:
        return filtered.sort_values('year').iloc[-1]['points']
    else:
        return 0

df['home_team_points'] = df.apply(lambda row: get_ranking_points(row['home_team'], row['date']), axis=1)
df['away_team_points'] = df.apply(lambda row: get_ranking_points(row['away_team'], row['date']), axis=1)

# Convert teams to numeric IDs
df['home_team_id'] = df['home_team'].astype('category').cat.codes
df['away_team_id'] = df['away_team'].astype('category').cat.codes

# Select features and target
feature_cols = [
    'home_team_id', 'away_team_id', 'neutral',
    'recent_home_wins', 'recent_home_draws', 'recent_home_losses',
    'recent_away_wins', 'recent_away_draws', 'recent_away_losses',
    'recent_home_goal_diff', 'recent_away_goal_diff',
    'home_team_points', 'away_team_points'
]

X = df[feature_cols]
y = df['result']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest and GridSearch (without SMOTE)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_test)

print(f"Best Params: {grid_search.best_params_}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


