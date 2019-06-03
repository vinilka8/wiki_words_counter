import unittest
import bag_of_words


class TestStringMethods(unittest.TestCase):

    def test_server_receive_message_from_client(self):
        web_data = bag_of_words.get_data_from_url(21721040)
        title = bag_of_words.get_title(web_data, 21721040)
        self.assertEqual(title, "Stack Overflow")
