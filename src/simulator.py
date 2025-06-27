import random
import numpy as np
import pandas as pd
from datetime import datetime
from .model import SelfHealingRacingModel
from .config import TRACKS, HORSE_POOL_SIZE, JOCKEY_POOL_SIZE, TRAINER_POOL_SIZE, SCRATCH_PROBABILITY

class RacingSimulator:
    def __init__(self):
        self.tracks = TRACKS
        self.horses = self.generate_horse_pool(HORSE_POOL_SIZE)
        self.jockeys = self.generate_jockey_pool(JOCKEY_POOL_SIZE)
        self.trainers = self.generate_trainer_pool(TRAINER_POOL_SIZE)
        self.model = SelfHealingRacingModel()
        self.scratch_log = []
        
    def generate_horse_pool(self, size):
        return [{
            'id': f"H_{i}",
            'name': f"Horse_{i}",
            'age': random.randint(3, 8),
            'base_speed': random.uniform(0.7, 1.0),
            'stamina': random.uniform(0.5, 0.95),
            'injury_risk': random.uniform(0.01, 0.25),
            'preferred_going': random.choice(["Good", "Soft", "Firm"]),
            'preferred_distance': random.choice([1000, 1200, 1600, 2000, 2400]),
            'last_races': [random.uniform(0.3, 1.0) for _ in range(3)]
        } for i in range(1, size+1)]
    
    def generate_jockey_pool(self, size):
        return [{
            'id': f"J_{i}",
            'name': f"Jockey_{i}",
            'win_rate': random.uniform(0.1, 0.4),
            'experience': random.randint(1, 15),
            'course_knowledge': {track: random.uniform(0.5, 1.0) for track in self.tracks}
        } for i in range(1, size+1)]
    
    def generate_trainer_pool(self, size):
        return [{
            'id': f"T_{i}",
            'name': f"Trainer_{i}",
            'win_rate': random.uniform(0.15, 0.45),
            'specialty': random.choice(["Sprinter", "Stayer", "All-Rounder"]),
            'form': random.uniform(0.6, 0.95)
        } for i in range(1, size+1)]
    
    def generate_race(self, track, race_time):
        distance = random.choice([1000, 1200, 1400, 1600, 2000, 2400])
        going = random.choice(["Good", "Good to Soft", "Soft", "Heavy", "Firm"])
        weather = random.choice(["Sunny", "Cloudy", "Light Rain", "Heavy Rain"])
        
        participants = random.sample(self.horses, random.randint(8, 16))
        race_data = []
        
        for horse in participants:
            jockey = random.choice(self.jockeys)
            trainer = random.choice(self.trainers)
            
            horse_data = {
                'horse_id': horse['id'],
                'horse': horse['name'],
                'jockey': jockey['name'],
                'trainer': trainer['name'],
                'horse_age': horse['age'],
                'jockey_win_rate': jockey['win_rate'] * jockey['course_knowledge'].get(track, 0.8),
                'trainer_win_rate': trainer['win_rate'],
                'weight': random.randint(50, 65),
                'distance_suitability': max(0.1, 1 - abs(distance - horse['preferred_distance'])/2000),
                'track_record': random.uniform(0.4, 0.95),
                'going_preference': 0.8 if going == horse['preferred_going'] else 0.5,
                'days_since_last_run': random.randint(14, 120),
                'recent_form': np.mean(horse['last_races']),
                'injury_risk': horse['injury_risk'],
                'news_sentiment': random.uniform(-0.3, 0.7),
                'course_specialty': jockey['course_knowledge'].get(track, 0.7),
                'class_drop': random.uniform(0.7, 1.2),
                'speed_rating': horse['base_speed'] * (1 - horse['age']/20),
                'stamina_index': horse['stamina'] * (distance/2000),
                'is_scratched': False,
                'scratch_reason': None
            }
            race_data.append(horse_data)
        
        return {
            'race_id': f"{track}_{datetime.now().strftime('%Y%m%d')}_{race_time.replace(':', '')}",
            'track': track,
            'time': race_time,
            'distance': distance,
            'going': going,
            'weather': weather,
            'participants': race_data,
            'last_scratch_check': datetime.now(),
            'scratch_updates': 0
        }
    
    def check_for_scratches(self, race):
        updated = False
        for horse in race['participants']:
            if horse['is_scratched']:
                continue
                
            scratch_prob = SCRATCH_PROBABILITY
            if horse['injury_risk'] > 0.15:
                scratch_prob *= 2
            if random.random() < scratch_prob:
                horse['is_scratched'] = True
                reasons = [
                    "Late veterinary concern",
                    "Travel issues",
                    "Owner decision",
                    "Ground conditions",
                    "Jockey illness"
                ]
                horse['scratch_reason'] = random.choice(reasons)
                self.scratch_log.append({
                    'race_id': race['race_id'],
                    'horse': horse['horse'],
                    'reason': horse['scratch_reason'],
                    'timestamp': datetime.now()
                })
                updated = True
                race['scratch_updates'] += 1
        return updated
    
    def generate_predictions(self, race):
        active_participants = [p for p in race['participants'] if not p['is_scratched']]
        
        if not active_participants:
            return pd.DataFrame()
            
        df = pd.DataFrame(active_participants)
        df['ai_win_prob'] = self.model.predict(df)
        
        # Bookmaker simulation
        df['hollywoodbets'] = (
            df['ai_win_prob'] * 
            (0.9 + 0.2 * df['trainer_win_rate']) * 
            (1.1 - 0.3 * df['injury_risk'])
        )
        
        df['betway'] = (
            df['ai_win_prob'] * 
            (0.85 + 0.15 * np.tanh(df['recent_form'] * 3)) *
            (1.05 - 0.2 * df['horse_age']/10)
        )
        
        # Normalize
        df['ai_win_prob'] = df['ai_win_prob'] / df['ai_win_prob'].sum()
        df['hollywoodbets'] = df['hollywoodbets'] / df['hollywoodbets'].sum()
        df['betway'] = df['betway'] / df['betway'].sum()
        
        return df.sort_values('ai_win_prob', ascending=False)
    
    def simulate_results(self, race):
        predictions = race.get('predictions', None)
        if predictions is None or len(predictions) == 0:
            return None
            
        winner_idx = np.random.choice(
            len(predictions), 
            p=predictions['ai_win_prob']
        )
        return winner_idx
    
    def update_model(self, race):
        results_df = pd.DataFrame(race['participants'])
        results_df['outcome'] = 0
        if race['result'] is not None:
            winner = predictions.iloc[race['result']]
            results_df.loc[results_df['horse'] == winner['horse'], 'outcome'] = 1
        
        self.model.add_results(results_df)
        if len(self.model.error_analysis) > 10:
            self.model.self_heal()
