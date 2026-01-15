"""
alerts.py - Alert State Machine + Persistence

Manages tiered alert levels with hysteresis and confirmation requirements.

Tiers:
- 0: Normal (Λ_geo < 2× baseline)
- 1: Watch (2× ≤ Λ_geo < 5×)
- 2: Elevated (5× ≤ Λ_geo < 10×, sustained 2+ days)
- 3: High (Λ_geo ≥ 10×, sustained + coherent)
"""

import json
import csv
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
from enum import IntEnum


class AlertTier(IntEnum):
    """Alert tier levels."""
    NORMAL = 0
    WATCH = 1
    ELEVATED = 2
    HIGH = 3


@dataclass
class AlertTransition:
    """Record of a tier transition."""
    timestamp: datetime
    from_tier: AlertTier
    to_tier: AlertTier
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyState:
    """Daily monitoring state for a region."""
    date: datetime
    region_id: str
    tier: AlertTier
    
    # Metrics
    lambda_max: float
    baseline_median: float
    ratio: float  # lambda_max / baseline
    zscore: float
    
    # Coherence
    is_coherent: bool
    cluster_size: int
    fraction_elevated: float
    
    # Baseline info
    baseline_n_days: int
    baseline_start: datetime
    baseline_end: datetime
    
    # Processing info
    version_hash: str = ""
    notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            'date': self.date.strftime('%Y-%m-%d'),
            'region': self.region_id,
            'tier': int(self.tier),
            'lambda_max': self.lambda_max,
            'baseline_median': self.baseline_median,
            'ratio': self.ratio,
            'zscore': self.zscore,
            'is_coherent': self.is_coherent,
            'cluster_size': self.cluster_size,
            'fraction_elevated': self.fraction_elevated,
            'baseline_n_days': self.baseline_n_days,
            'baseline_start': self.baseline_start.strftime('%Y-%m-%d'),
            'baseline_end': self.baseline_end.strftime('%Y-%m-%d'),
            'version_hash': self.version_hash,
            'notes': self.notes,
        }
    
    def to_csv_row(self) -> dict:
        """Return dict suitable for CSV writing."""
        return {
            'date': self.date.strftime('%Y-%m-%d'),
            'region': self.region_id,
            'tier': int(self.tier),
            'amp_72h': f"{self.ratio:.2f}",
            'zscore': f"{self.zscore:.2f}",
            'coherent': str(self.is_coherent).lower(),
            'fpr_5x': "",  # Computed separately
            'notes': self.notes,
            'version_hash': self.version_hash[:8] if self.version_hash else "",
        }


class AlertStateMachine:
    """
    Manages alert tier transitions with hysteresis.
    
    Transition rules:
    - Escalation to Tier 2+ requires 2 consecutive days
    - De-escalation requires 3 consecutive days below threshold
    - Tier 3 requires spatial coherence
    """
    
    # Thresholds
    TIER_1_THRESHOLD = 2.0   # ≥ 2× baseline
    TIER_2_THRESHOLD = 5.0   # ≥ 5× baseline
    TIER_3_THRESHOLD = 10.0  # ≥ 10× baseline
    
    # Persistence requirements
    ESCALATION_DAYS = 2      # Days to confirm escalation to Tier 2+
    DEESCALATION_DAYS = 3    # Days to confirm de-escalation
    
    def __init__(self, region_id: str):
        self.region_id = region_id
        self.current_tier = AlertTier.NORMAL
        self.tier_entry_date: Optional[datetime] = None
        self.consecutive_days_at_tier = 0
        self.consecutive_days_below = 0
        self.pending_escalation: Optional[AlertTier] = None
        self.pending_days = 0
        
        self.history: List[DailyState] = []
        self.transitions: List[AlertTransition] = []
    
    def update(self, 
               date: datetime,
               lambda_max: float,
               baseline_median: float,
               zscore: float,
               is_coherent: bool,
               cluster_size: int = 0,
               fraction_elevated: float = 0.0,
               baseline_n_days: int = 0,
               baseline_start: Optional[datetime] = None,
               baseline_end: Optional[datetime] = None,
               version_hash: str = "") -> AlertTier:
        """
        Update alert state with new observation.
        
        Returns new tier level.
        """
        ratio = lambda_max / baseline_median if baseline_median > 0 else 0
        
        # Determine candidate tier based on thresholds
        candidate = self._compute_candidate_tier(ratio, is_coherent)
        
        # Apply persistence logic
        new_tier = self._apply_persistence(candidate, date)
        
        # Record transition if tier changed
        if new_tier != self.current_tier:
            transition = AlertTransition(
                timestamp=date,
                from_tier=self.current_tier,
                to_tier=new_tier,
                reason=self._get_transition_reason(candidate, new_tier),
                metrics={'ratio': ratio, 'zscore': zscore, 'coherent': is_coherent}
            )
            self.transitions.append(transition)
            self.current_tier = new_tier
            self.tier_entry_date = date
            self.consecutive_days_at_tier = 1
        else:
            self.consecutive_days_at_tier += 1
        
        # Record daily state
        state = DailyState(
            date=date,
            region_id=self.region_id,
            tier=self.current_tier,
            lambda_max=lambda_max,
            baseline_median=baseline_median,
            ratio=ratio,
            zscore=zscore,
            is_coherent=is_coherent,
            cluster_size=cluster_size,
            fraction_elevated=fraction_elevated,
            baseline_n_days=baseline_n_days,
            baseline_start=baseline_start or date,
            baseline_end=baseline_end or date,
            version_hash=version_hash,
        )
        self.history.append(state)
        
        return self.current_tier
    
    def _compute_candidate_tier(self, ratio: float, is_coherent: bool) -> AlertTier:
        """Compute candidate tier from current metrics."""
        if ratio >= self.TIER_3_THRESHOLD and is_coherent:
            return AlertTier.HIGH
        elif ratio >= self.TIER_2_THRESHOLD:
            return AlertTier.ELEVATED
        elif ratio >= self.TIER_1_THRESHOLD:
            return AlertTier.WATCH
        else:
            return AlertTier.NORMAL
    
    def _apply_persistence(self, candidate: AlertTier, date: datetime) -> AlertTier:
        """Apply persistence/hysteresis rules."""
        
        # Case 1: Same tier as current - no change
        if candidate == self.current_tier:
            self.pending_escalation = None
            self.pending_days = 0
            self.consecutive_days_below = 0
            return self.current_tier
        
        # Case 2: Escalation
        if candidate > self.current_tier:
            self.consecutive_days_below = 0
            
            # Tier 0→1 is immediate
            if self.current_tier == AlertTier.NORMAL and candidate == AlertTier.WATCH:
                return AlertTier.WATCH
            
            # Tier 2+ requires confirmation
            if self.pending_escalation == candidate:
                self.pending_days += 1
                if self.pending_days >= self.ESCALATION_DAYS:
                    self.pending_escalation = None
                    self.pending_days = 0
                    return candidate
            else:
                self.pending_escalation = candidate
                self.pending_days = 1
            
            return self.current_tier
        
        # Case 3: De-escalation
        if candidate < self.current_tier:
            self.pending_escalation = None
            self.pending_days = 0
            self.consecutive_days_below += 1
            
            if self.consecutive_days_below >= self.DEESCALATION_DAYS:
                self.consecutive_days_below = 0
                return candidate
            
            return self.current_tier
        
        return self.current_tier
    
    def _get_transition_reason(self, candidate: AlertTier, actual: AlertTier) -> str:
        """Generate human-readable transition reason."""
        if actual > self.current_tier:
            return f"Escalated to Tier {actual.value} after sustained elevation"
        elif actual < self.current_tier:
            return f"De-escalated to Tier {actual.value} after {self.DEESCALATION_DAYS} days below threshold"
        return "No change"
    
    def get_latest_state(self) -> Optional[DailyState]:
        """Get most recent daily state."""
        return self.history[-1] if self.history else None
    
    def get_tier_description(self) -> str:
        """Get human-readable tier description."""
        descriptions = {
            AlertTier.NORMAL: "Normal - No anomaly detected",
            AlertTier.WATCH: "Watch - Elevated signal (2-5×)",
            AlertTier.ELEVATED: "Elevated - Significant anomaly (5-10×)",
            AlertTier.HIGH: "High - Major anomaly (10×+, coherent)",
        }
        return descriptions.get(self.current_tier, "Unknown")


class AlertStorage:
    """
    Persistent storage for alert states and history.
    
    Supports both CSV (append-only log) and SQLite (queryable).
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.data_dir / "daily_states.csv"
        self.db_path = self.data_dir / "monitoring.db"
        
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                region TEXT NOT NULL,
                tier INTEGER NOT NULL,
                lambda_max REAL,
                baseline_median REAL,
                ratio REAL,
                zscore REAL,
                is_coherent INTEGER,
                cluster_size INTEGER,
                fraction_elevated REAL,
                baseline_n_days INTEGER,
                baseline_start TEXT,
                baseline_end TEXT,
                version_hash TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, region)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                region TEXT NOT NULL,
                from_tier INTEGER,
                to_tier INTEGER,
                reason TEXT,
                metrics TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_states_region_date 
            ON daily_states(region, date)
        """)
        
        conn.commit()
        conn.close()
    
    def save_state(self, state: DailyState):
        """Save daily state to both CSV and SQLite."""
        self._save_to_csv(state)
        self._save_to_db(state)
    
    def _save_to_csv(self, state: DailyState):
        """Append state to CSV (idempotent)."""
        # Check if row already exists
        existing_keys = set()
        if self.csv_path.exists():
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_keys.add((row['date'], row['region']))
        
        key = (state.date.strftime('%Y-%m-%d'), state.region_id)
        if key in existing_keys:
            return  # Already exists
        
        # Append row
        write_header = not self.csv_path.exists()
        fieldnames = ['date', 'region', 'tier', 'amp_72h', 'zscore', 
                      'coherent', 'fpr_5x', 'notes', 'version_hash']
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(state.to_csv_row())
    
    def _save_to_db(self, state: DailyState):
        """Save state to SQLite (upsert)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO daily_states 
            (date, region, tier, lambda_max, baseline_median, ratio, zscore,
             is_coherent, cluster_size, fraction_elevated, baseline_n_days,
             baseline_start, baseline_end, version_hash, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.date.strftime('%Y-%m-%d'),
            state.region_id,
            int(state.tier),
            state.lambda_max,
            state.baseline_median,
            state.ratio,
            state.zscore,
            int(state.is_coherent),
            state.cluster_size,
            state.fraction_elevated,
            state.baseline_n_days,
            state.baseline_start.strftime('%Y-%m-%d'),
            state.baseline_end.strftime('%Y-%m-%d'),
            state.version_hash,
            state.notes,
        ))
        
        conn.commit()
        conn.close()
    
    def save_transition(self, region_id: str, transition: AlertTransition):
        """Save alert transition to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alert_transitions
            (timestamp, region, from_tier, to_tier, reason, metrics)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            transition.timestamp.isoformat(),
            region_id,
            int(transition.from_tier),
            int(transition.to_tier),
            transition.reason,
            json.dumps(transition.metrics),
        ))
        
        conn.commit()
        conn.close()
    
    def get_latest_states(self) -> Dict[str, DailyState]:
        """Get latest state for each region."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM daily_states 
            WHERE (region, date) IN (
                SELECT region, MAX(date) FROM daily_states GROUP BY region
            )
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to DailyState objects (simplified)
        result = {}
        for row in rows:
            region = row[2]  # Assuming column order
            result[region] = row
        
        return result
    
    def get_region_history(self, region_id: str, 
                           days: int = 30) -> List[Dict]:
        """Get recent history for a region."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM daily_states 
            WHERE region = ?
            ORDER BY date DESC
            LIMIT ?
        """, (region_id, days))
        
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return rows


# === Tests ===

def test_state_machine():
    """Test alert state machine transitions."""
    sm = AlertStateMachine("test_region")
    
    baseline = 0.1
    
    # Day 1: Normal
    tier = sm.update(datetime(2024, 1, 1), lambda_max=0.15, 
                     baseline_median=baseline, zscore=0.5, is_coherent=False)
    assert tier == AlertTier.NORMAL
    print(f"Day 1: Tier {tier.value} (ratio 1.5×)")
    
    # Day 2: Watch (2.5×)
    tier = sm.update(datetime(2024, 1, 2), lambda_max=0.25,
                     baseline_median=baseline, zscore=1.5, is_coherent=False)
    assert tier == AlertTier.WATCH
    print(f"Day 2: Tier {tier.value} (ratio 2.5×)")
    
    # Day 3-4: Elevated but needs confirmation
    tier = sm.update(datetime(2024, 1, 3), lambda_max=0.6,
                     baseline_median=baseline, zscore=5.0, is_coherent=True)
    assert tier == AlertTier.WATCH  # Pending confirmation
    print(f"Day 3: Tier {tier.value} (ratio 6×, pending)")
    
    tier = sm.update(datetime(2024, 1, 4), lambda_max=0.7,
                     baseline_median=baseline, zscore=6.0, is_coherent=True)
    assert tier == AlertTier.ELEVATED  # Confirmed
    print(f"Day 4: Tier {tier.value} (ratio 7×, confirmed)")
    
    # Day 5: High (11×, coherent)
    tier = sm.update(datetime(2024, 1, 5), lambda_max=1.1,
                     baseline_median=baseline, zscore=10.0, is_coherent=True)
    print(f"Day 5: Tier {tier.value} (ratio 11×, pending)")
    
    tier = sm.update(datetime(2024, 1, 6), lambda_max=1.2,
                     baseline_median=baseline, zscore=11.0, is_coherent=True)
    assert tier == AlertTier.HIGH
    print(f"Day 6: Tier {tier.value} (ratio 12×, confirmed)")
    
    # Day 7-9: De-escalation
    for i in range(3):
        tier = sm.update(datetime(2024, 1, 7 + i), lambda_max=0.15,
                         baseline_median=baseline, zscore=0.5, is_coherent=False)
        print(f"Day {7+i}: Tier {tier.value} (ratio 1.5×)")
    
    assert tier == AlertTier.NORMAL
    print("\n✓ State machine test passed")
    print(f"Transitions: {len(sm.transitions)}")


if __name__ == "__main__":
    test_state_machine()
