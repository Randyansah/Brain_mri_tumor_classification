from main import tf_version
import tensorflow as tf

def test_add():
    assert "2.13.0" in tf_version(tf.version.VERSION)