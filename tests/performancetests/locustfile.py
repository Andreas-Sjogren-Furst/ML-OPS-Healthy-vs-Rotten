"""
Module for performance tests using Locust.
"""

from time import sleep
from locust import HttpUser, task, between


class QuickstartUser(HttpUser):
    """
    A Locust user class that simulates behavior of users performing tasks on a system.
    """

    wait_time = between(1, 5)

    @task
    def hello_world(self):
        """
        Simulates a user visiting the /hello and /world endpoints.
        """
        self.client.get("/hello")
        self.client.get("/world")

    @task(3)
    def view_items(self):
        """
        Simulates a user viewing multiple items, sending requests to /item endpoint.
        """
        for item_id in range(10):
            self.client.get(f"/item?id={item_id}", name="/item")
            sleep(1)

    def on_start(self):
        """
        Simulates the login behavior of a user when the test starts.
        """
        self.client.post("/login", json={"username": "foo", "password": "bar"})
