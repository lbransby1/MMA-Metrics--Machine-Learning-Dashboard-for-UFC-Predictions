import numpy as pd

def create_fight(fighter_1, fighter_2, fighters_df):
    def create_features_from_df(f1, f2, df):
        red = df[df["Name"] == f1].drop(columns=["Name", "DOB"], errors='ignore').iloc[0]
        blue = df[df["Name"] == f2].drop(columns=["Name", "DOB"], errors='ignore').iloc[0]
        red.index = ["red_" + col for col in red.index]
        blue.index = ["blue_" + col for col in blue.index]
        return pd.concat([red, blue]).to_frame().T


    original_X = create_features_from_df(fighter_1, fighter_2, fighters_df)
    swapped_X = create_features_from_df(fighter_2, fighter_1, fighters_df)
    numeric_cols = [
        'red_StrikesLandedPerMin', 'red_StrikesAbsorbedPerMin', 'red_TakedownsPer15Min', 'red_SubmissionsPer15Min',
        'red_ControlPer15Min', 'red_StrikingAccuracyPct', 'red_StrikeDefencePct', 'red_TakedownAccuracyPct', 'red_TakedownDefencePct',
        'blue_StrikesLandedPerMin', 'blue_StrikesAbsorbedPerMin', 'blue_TakedownsPer15Min', 'blue_SubmissionsPer15Min',
        'blue_ControlPer15Min', 'blue_StrikingAccuracyPct', 'blue_StrikeDefencePct', 'blue_TakedownAccuracyPct', 'blue_TakedownDefencePct',
        'red_OpponentTakedownsPer15Min', 'blue_OpponentTakedownsPer15Min',
        'red_Height', 'red_Weight', 'red_Reach', 'blue_Height', 'blue_Weight', 'blue_Reach', 'red_Elo', 'blue_Elo', 'red_Age', 'blue_Age']

    for col in numeric_cols:
        if col in original_X.columns:
            original_X[col] = pd.to_numeric(original_X[col], errors='coerce')
            swapped_X[col] = pd.to_numeric(swapped_X[col], errors='coerce')

    categorical_cols = ["red_Stance", "blue_Stance"]
    for col in categorical_cols:
        original_X[col] = original_X[col].astype("category")
        swapped_X[col] = swapped_X[col].astype("category")
    
    return original_X, swapped_X