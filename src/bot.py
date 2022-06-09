#!/usr/bin/env python
# pylint: disable=C0116,W0613
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
pip install python-telegram-bot --upgrade
"""


#telegram init
import logging

from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

import gpt2forbot

current_model = "run1dialogs"
gpt2 = gpt2forbot.GPT2(model_name=current_model)



# Different constants for this example
(
    Transcript,
    Turns,
    GENDER,
    AGE,
    START_OVER,
) = map(chr, range(70, 75))

# Enable logging
logging.basicConfig(
    filename='bot.log', filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)
logging.info('Bot started')


def prepare_transcript(trans_list, turns_list):
    text = ""
    for i in range(0, len(turns_list)):
        if(turns_list[i]==0): text += "Me: "
        else: text += "You: "
        text += trans_list[i] + "\n"
    #print(text)
    return text

def strip_gpt2_covo(answer):
    lines = answer.split("\n")
    line = ""
    for line in lines:
      if (line !="" and ("Me:" in line)):
        line = line.split("Me:")[1]
        return line
        break
    return ""



# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    first_message = "Hi " + user.first_name + "!"
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )
    context.user_data[START_OVER] = False
    context.user_data[Transcript] = [first_message]
    context.user_data[Turns] = [0]


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""

    if((Transcript not in context.user_data) or (len(context.user_data[Transcript]) < 1)):
        update.message.reply_text("I have forgotten what we talked about before. Let's start over.")
        start(update, context)
        return
    context.user_data[Transcript].append(update.message.text)
    context.user_data[Turns].append(1)

    print("Beep")
    answer = ""
    while (len(answer)<2):
        answer = gpt2.generate_conditional(raw_text=prepare_transcript(context.user_data[Transcript], context.user_data[Turns]), last_length=len(update.message.text))
        answer = strip_gpt2_covo(answer)

        while((len(answer)>1) and (answer[-1]=='.' or answer[-1]=='?' or answer[-1]=='!')==False):
            if('. ' in answer):
                answer = answer.split('. ')[0] + '.'
            elif('?' in answer):
                answer = answer.split('?')[0] + '?'
            elif('!' in answer):
                answer = answer.split('!')[0] + '!'
            elif('[' in answer):
                answer = answer.split('[')[0]
            else:
                print("beep", answer)
                more = gpt2.generate_conditional(answer, last_length=len(update.message.text))
                
                answer += ' '
                answer += more
    print("Boop")

    update.message.reply_text(answer)
    context.user_data[Transcript].append(answer)
    context.user_data[Turns].append(0)

    #print(prepare_transcript(context.user_data[Transcript], context.user_data[Turns]))

    logging.info(update.effective_user.first_name + ': ' + update.message.text + '\tBOT: ' + answer)

    gpt2.generate_conditional


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("5402665772:AAEFa6yZ03Mef-5XsA2-qQSthPwsU7_fZ8E")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
