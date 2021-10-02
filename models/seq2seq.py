import argparse

import tensorflow as tf


class Seq2Seq(tf.keras.Model):
    
    @staticmethod
    def parseArgument(parser : argparse.ArgumentParser):

        parser.add_argument("--attention_type", type=str, default="ScaledDot", help="one of {'Dot', 'ScaledDot', 'Bahdanau'}")
        parser.add_argument("--attention_unit_num", type=int, default=64)
        