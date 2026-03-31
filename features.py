import pandas as pd

# Normalize data into kaggle format
def normalize_games(df, gender):
    df = df.copy()
    
    # Always lower ID first
    df["Team1"] = df[["WTeamID", "LTeamID"]].min(axis=1)
    df["Team2"] = df[["WTeamID", "LTeamID"]].max(axis=1)
    
    # Binary outcome from Team1 perspective
    df["Team1Win"] = (df["Team1"] == df["WTeamID"]).astype(int)
    
    # Raw margin (winner perspective)
    df["RawPointDiff"] = df["WScore"] - df["LScore"]
    
    # Convert to Team1 perspective
    df["PointDiff"] = df["RawPointDiff"]
    df.loc[df["Team1Win"] == 0, "PointDiff"] *= -1
    
    df["Gender"] = gender
    
    return df[[
        "Season",
        "DayNum",
        "Team1",
        "Team2",
        "Team1Win",
        "PointDiff",
        "Gender"
    ]]

# Build stats dataset
def build_team_season_stats(games_df):
    
    # Build perspective for Team1
    team1_stats = games_df.groupby(
        ["Season", "Team1"]
    ).agg(
        Games=("Team1Win", "count"),
        Wins=("Team1Win", "sum"),
        AvgPointDiff=("PointDiff", "mean")
    ).reset_index().rename(columns={"Team1": "TeamID"})
    
    # Build perspective for Team2
    team2_df = games_df.copy()
    team2_df["Team2Win"] = 1 - team2_df["Team1Win"]
    team2_df["Team2PointDiff"] = -team2_df["PointDiff"]
    
    team2_stats = team2_df.groupby(
        ["Season", "Team2"]
    ).agg(
        Games=("Team2Win", "count"),
        Wins=("Team2Win", "sum"),
        AvgPointDiff=("Team2PointDiff", "mean")
    ).reset_index().rename(columns={"Team2": "TeamID"})
    
    # Combine both perspectives
    all_team_stats = pd.concat([team1_stats, team2_stats])
    
    # Final aggregation (because teams appear in both sets)
    final_stats = all_team_stats.groupby(
        ["Season", "TeamID"]
    ).agg(
        Games=("Games", "sum"),
        Wins=("Wins", "sum"),
        AvgPointDiff=("AvgPointDiff", "mean")
    ).reset_index()
    
    final_stats["WinPct"] = final_stats["Wins"] / final_stats["Games"]
    
    return final_stats

# Build dataset for matchup stats
def build_matchup_dataset(games_df, team_stats):
    
    # Merge Team1 stats
    df = games_df.merge(
        team_stats,
        left_on=["Season", "Team1"],
        right_on=["Season", "TeamID"],
        how="left"
    ).rename(columns={
        "WinPct": "Team1_WinPct",
        "AvgPointDiff": "Team1_AvgPD"
    }).drop(columns=["TeamID"])
    
    # Merge Team2 stats
    df = df.merge(
        team_stats,
        left_on=["Season", "Team2"],
        right_on=["Season", "TeamID"],
        how="left"
    ).rename(columns={
        "WinPct": "Team2_WinPct",
        "AvgPointDiff": "Team2_AvgPD"
    }).drop(columns=["TeamID"])
    
    # Feature engineering
    df["WinPctDiff"] = df["Team1_WinPct"] - df["Team2_WinPct"]
    df["AvgPDDiff"] = df["Team1_AvgPD"] - df["Team2_AvgPD"]
    
    # Keep modeling columns
    model_df = df[[
        "Season",
        "Team1",
        "Team2",
        "WinPctDiff",
        "AvgPDDiff",
        "Team1Win"
    ]]
    
    return model_df

def add_seed_features(matchups_df, seeds_df):
    
    df = matchups_df.merge(
        seeds_df[["Season", "TeamID", "SeedNum"]],
        left_on=["Season", "Team1"],
        right_on=["Season", "TeamID"],
        how="left"
    ).rename(columns={"SeedNum": "Team1Seed"}).drop(columns=["TeamID"])
    
    df = df.merge(
        seeds_df[["Season", "TeamID", "SeedNum"]],
        left_on=["Season", "Team2"],
        right_on=["Season", "TeamID"],
        how="left"
    ).rename(columns={"SeedNum": "Team2Seed"}).drop(columns=["TeamID"])
    
    df["SeedDiff"] = df["Team2Seed"] - df["Team1Seed"]
    
    return df

def compute_elo_ratings(df, k=20, base_elo=1500):
    """
    Computes Elo ratings per season using regular season games only.
    Returns:
        elo_df: DataFrame with final pre-tournament Elo per team per season
    """
    
    df = df.sort_values(["Season", "DayNum"]).copy()
    
    elo_dict = {}
    final_elos = []

    for season in df["Season"].unique():
        season_df = df[df["Season"] == season]
        
        # Initialize ratings
        teams = pd.unique(
            season_df[["WTeamID", "LTeamID"]].values.ravel()
        )
        ratings = {team: base_elo for team in teams}
        
        for _, row in season_df.iterrows():
            w = row["WTeamID"]
            l = row["LTeamID"]
            
            Ra = ratings[w]
            Rb = ratings[l]
            
            # Expected win probability
            Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
            Eb = 1 - Ea
            
            # Update ratings
            ratings[w] = Ra + k * (1 - Ea)
            ratings[l] = Rb + k * (0 - Eb)
        
        # Store final ratings
        for team, rating in ratings.items():
            final_elos.append({
                "Season": season,
                "TeamID": team,
                "Elo": rating
            })
    
    elo_df = pd.DataFrame(final_elos)
    return elo_df

