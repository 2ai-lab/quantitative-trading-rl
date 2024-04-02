from enum import Enum

class Action(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2

action_space = [action.value for action in Action]

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, state_space=3, action_space=len(action_space)):
        self.data = data
        self.initial_balance = initial_balance
        
        # Properties
        self.current_step = 0
        self.balance = initial_balance
        self.stock_owned = 0
        self.net_worth = initial_balance
        self.last_trade_index = self.current_step - 1
        
        # buy, sell, hold
        self.action_space = action_space
        # state components: [balance, owned_stock_price, current_price]
        self.state_size = state_space

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.net_worth = self.initial_balance
        self.done = False
        self.last_trade_index = self.current_step - 1
        return self._get_state()

    def _get_state(self):
        # Fetches the current state of the environment
        info = self.data.loc[self.current_step]
        current_price = info['Close']
        owned_stock_price = self.stock_owned * current_price
        state = [self.balance, owned_stock_price, current_price]
        return state

    def step(self, action):
        self.current_step += 1
        done = self.current_step == len(self.data) - 1
        
        current_price = self.data.loc[self.current_step, 'Close']
        self.net_worth = self.balance + self.stock_owned * current_price

        # Take the action: buy, sell, or hold
        if action == 0 and self.balance > current_price:  # buy
            self.stock_owned += 1
            self.balance -= current_price
        elif action == 1 and self.stock_owned > 0:  # sell
            self.stock_owned -= 1
            self.balance += current_price
        elif action == 2:  # hold
            pass

        # Calculate reward
        earnings = self.balance + (self.stock_owned * current_price) - self.initial_balance
        reward = earnings / self.initial_balance if earnings > 0 else -0.1

        # Prepare the state for the next time step
        state = self._get_state()
        labels = self.data.loc[self.current_step, 'Action']

        return state, reward, done, labels