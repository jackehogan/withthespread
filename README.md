# withthespread

## Predict NFL spreads based on past weeks performance against the spread
### Routine runs weekly

Routine Description
1. past weeks game results are pulled with NFL API and compared with the previous weeks predicitons
2.Model (currently XGBoost) is trained on NFL season data from 2000+ 
3. Next weeks spreads are pulled from with API call, and using past weeks spreads predictions are made.
4. Store new predictions in MongoDB

Next steps are visualizing results, will wait until further into season for better dataset.

