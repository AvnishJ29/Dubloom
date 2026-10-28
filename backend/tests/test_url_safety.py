"""A public link may not redirect a download to a local-network address."""

import socket
import unittest
from unittest.mock import patch

import httpx

from backend.media import video_input
from backend.media.video_input import SourceError

PUBLIC = "93.184.216.34"
REAL_CLIENT = httpx.Client  # captured before the tests patch httpx.Client


def resolve(host, *args, **kwargs):
    address = "127.0.0.1" if host in {"127.0.0.1", "intranet.example"} else PUBLIC
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]


def site(request):
    if request.url.host == "public.example":
        return httpx.Response(302, headers={"location": "http://intranet.example/admin.mp4"})
    return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"secret")


def mock_client(**kwargs):
    return REAL_CLIENT(transport=httpx.MockTransport(site), **kwargs)


class RedirectTest(unittest.TestCase):
    def test_redirect_to_a_local_address_is_refused(self):
        with (
            patch.object(video_input.socket, "getaddrinfo", resolve),
            patch.object(video_input.httpx, "Client", mock_client),
        ):
            with self.assertRaisesRegex(SourceError, "Local network"):
                video_input._inspect_url("http://public.example/video")

    def test_public_links_still_work(self):
        with (
            patch.object(video_input.socket, "getaddrinfo", resolve),
            patch.object(video_input.httpx, "Client", mock_client),
        ):
            content_type, _, final = video_input._inspect_url("http://cdn.example/clip.mp4")
        self.assertEqual(content_type, "video/mp4")
        self.assertEqual(final, "http://cdn.example/clip.mp4")


if __name__ == "__main__":
    unittest.main()
