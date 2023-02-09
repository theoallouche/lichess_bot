===========
Lichess bot
===========


| Pure Python 3 package aiming to run a chess bot on `Lichess <https://lichess.org/>`_ platform.
| It uses the Negamax tree-search algorithm and a very simple evaluation function only based on material value.



Installation
============

Just install the (2) dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Then, you will need a Lichess bot account, which consists in creating a dedicated account, and getting an `API token <https://lichess.org/account/oauth/token/create>`_ .


Getting started
~~~~~~~~~~~~~~~

.. code-block:: python

   from api_lichess import LichessBot

   token = " Your token..."
   client = LichessBot(token_session=token)
   client.launch()

Once it is launched, everyone should be able to challenge it online.

Feel free to change the depth search, customize the evaluation function and change the timeout before playing a move.
