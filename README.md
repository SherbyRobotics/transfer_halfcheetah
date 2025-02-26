# Study on policy transfer on the Half Cheetah

The base policy used in this study is available [here](https://huggingface.co/farama-minari/HalfCheetah-v5-SAC-expert).

The transfer is done using [Pipoli](https://github.com/SherbyRobotics/pipoli).

## Run it

1. Clone this repo and open a terminal in its directory.

2. Create a virtual environment and activate it.
   ```sh
   python3 -m venv .venv && . .venv/bin/activate
   ```

3. Install the requirements
   ```sh
   python3 -m pip install -r requirements.txt
   ```

4. Run!
   ```
   python3 transfer.py
   ```