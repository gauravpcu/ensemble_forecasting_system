#!/usr/bin/env python3
"""
Calibration Manager
Centralized management of customer-specific calibrations
"""

import json
import os
from typing import Dict, Optional, Tuple
from datetime import datetime

class CalibrationManager:
    """Manage customer-specific calibrations"""
    
    def __init__(self, config_file: str = 'config/customer_calibrations.json'):
        self.config_file = config_file
        self._data = None
    
    def load(self) -> Dict:
        """Load calibration data"""
        if self._data is not None:
            return self._data
        
        if not os.path.exists(self.config_file):
            # Return default configuration
            self._data = {
                "version": "1.0",
                "last_updated": datetime.now().strftime('%Y-%m-%d'),
                "default_calibration": 1.0,
                "classification_threshold": 4,
                "customers": {}
            }
            return self._data
        
        with open(self.config_file, 'r') as f:
            self._data = json.load(f)
        
        return self._data
    
    def get_calibration(self, customer_id: str) -> float:
        """
        Get calibration multiplier for a customer
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Calibration multiplier (default 1.0 if not found)
        """
        data = self.load()
        customer_id = customer_id.lower()
        
        if customer_id in data['customers']:
            return data['customers'][customer_id].get('calibration_multiplier', 1.0)
        
        return data.get('default_calibration', 1.0)
    
    def get_customer_config(self, customer_id: str) -> Dict:
        """
        Get full configuration for a customer
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Customer configuration dictionary
        """
        data = self.load()
        customer_id = customer_id.lower()
        
        if customer_id in data['customers']:
            return data['customers'][customer_id]
        
        return {
            'calibration_multiplier': data.get('default_calibration', 1.0),
            'status': 'unknown',
            'notes': 'No configuration found. Using default calibration.'
        }
    
    def is_production_ready(self, customer_id: str) -> Tuple[bool, str]:
        """
        Check if customer configuration is production ready
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Tuple of (is_ready, message)
        """
        config = self.get_customer_config(customer_id)
        status = config.get('status', 'unknown')
        
        if status == 'verified':
            return True, "Configuration verified and production ready"
        elif status == 'needs_recalibration':
            return False, f"Needs recalibration. {config.get('recommendation', 'Review configuration.')}"
        elif status == 'testing':
            return False, "Currently in testing. Not ready for production."
        elif status == 'deprecated':
            return False, "Configuration deprecated. Update required."
        else:
            return False, "Configuration not verified. Test before deployment."
    
    def get_threshold(self) -> int:
        """Get classification threshold"""
        data = self.load()
        return data.get('classification_threshold', 4)
    
    def get_all_calibrations(self) -> Dict[str, float]:
        """
        Get all customer calibrations as a dictionary
        
        Returns:
            Dictionary mapping customer_id to calibration multiplier
        """
        data = self.load()
        calibrations = {}
        
        for customer_id, config in data['customers'].items():
            calibrations[customer_id] = config.get('calibration_multiplier', 1.0)
        
        return calibrations
    
    def update_metrics(
        self,
        customer_id: str,
        precision: float,
        recall: float,
        mae: float,
        status: Optional[str] = None
    ):
        """
        Update performance metrics for a customer
        
        Args:
            customer_id: Customer identifier
            precision: Precision score (0-1)
            recall: Recall score (0-1)
            mae: Mean Absolute Error
            status: Optional status update
        """
        data = self.load()
        customer_id = customer_id.lower()
        
        if customer_id not in data['customers']:
            data['customers'][customer_id] = {
                'calibration_multiplier': data.get('default_calibration', 1.0)
            }
        
        data['customers'][customer_id]['precision'] = precision
        data['customers'][customer_id]['recall'] = recall
        data['customers'][customer_id]['mae'] = mae
        data['customers'][customer_id]['last_tested'] = datetime.now().strftime('%Y-%m-%d')
        
        if status:
            data['customers'][customer_id]['status'] = status
        
        # Auto-determine status based on metrics
        if precision >= 0.8 and recall >= 0.5:
            data['customers'][customer_id]['status'] = 'verified'
        elif precision < 0.5 or recall < 0.3:
            data['customers'][customer_id]['status'] = 'needs_recalibration'
        
        self._save(data)
    
    def _save(self, data: Dict):
        """Save calibration data"""
        data['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._data = data

# Global instance
_manager = None

def get_manager() -> CalibrationManager:
    """Get global calibration manager instance"""
    global _manager
    if _manager is None:
        _manager = CalibrationManager()
    return _manager

def get_calibration(customer_id: str) -> float:
    """Convenience function to get calibration"""
    return get_manager().get_calibration(customer_id)

def is_production_ready(customer_id: str) -> Tuple[bool, str]:
    """Convenience function to check production readiness"""
    return get_manager().is_production_ready(customer_id)

if __name__ == "__main__":
    # Test the manager
    manager = CalibrationManager()
    
    print("Testing Calibration Manager")
    print("=" * 80)
    
    # Test ScionHealth
    print("\nScionHealth:")
    print(f"  Calibration: {manager.get_calibration('scionhealth')}")
    ready, msg = manager.is_production_ready('scionhealth')
    print(f"  Production Ready: {ready}")
    print(f"  Message: {msg}")
    
    # Test Mercy
    print("\nMercy:")
    print(f"  Calibration: {manager.get_calibration('mercy')}")
    ready, msg = manager.is_production_ready('mercy')
    print(f"  Production Ready: {ready}")
    print(f"  Message: {msg}")
    
    # Test unknown customer
    print("\nUnknown Customer:")
    print(f"  Calibration: {manager.get_calibration('unknown')}")
    ready, msg = manager.is_production_ready('unknown')
    print(f"  Production Ready: {ready}")
    print(f"  Message: {msg}")
