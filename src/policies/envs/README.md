# Reward


```
    def _reward_function(self):
        # Reward is the change of balance
        pnl = self.account.balance - self.balance
        reward = pnl / self.balance
        self.balance = deepcopy(self.account.balance)

        self.accumulated_reward += reward
        return reward
```