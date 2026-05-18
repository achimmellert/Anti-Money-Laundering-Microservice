import flet as ft
import httpx
import csv
import asyncio
from pathlib import Path
import os
from datetime import datetime


API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parents[2]
CSV_PATH = BASE_DIR / "data" / "test.csv"

http_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=500, max_keepalive_connections=100))


class AMLDashboard:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "AML Fraud Detection - Dashboard"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 20

        self.poll_semaphore = asyncio.Semaphore(10)

        self.setup_ui()

    def setup_ui(self):
        # --- VIEW 1: Live Monitor ---
        self.start_button = ft.Button("Test-Transaktionen senden", on_click=self.start_tests, icon=ft.Icons.PLAY_ARROW,
                                      style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700))
        self.clear_button = ft.Button("Dashboard leeren", on_click=self.clear_dashboard, icon=ft.Icons.DELETE_SWEEP,
                                      style=ft.ButtonStyle(color=ft.Colors.RED_400))
        self.progress_text = ft.Text("Warte auf Start...", italic=True)
        self.data_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Event ID")),
                ft.DataColumn(ft.Text("Sender")),
                ft.DataColumn(ft.Text("Empfänger")),
                ft.DataColumn(ft.Text("Betrag", text_align=ft.TextAlign.RIGHT)),
                ft.DataColumn(ft.Text("Status")),
                ft.DataColumn(ft.Text("Fraud Score", text_align=ft.TextAlign.RIGHT)),
            ],
            rows=[]
        )
        self.view_monitor = ft.Container(
            content=ft.Column([
                ft.Row([self.start_button, self.clear_button, self.progress_text]),
                ft.Column([self.data_table], scroll=ft.ScrollMode.ADAPTIVE, expand=True)
            ], expand=True),
            key="monitor", expand=True
        )

        # --- VIEW 2: User Search ---
        self.search_input = ft.TextField(label="User ID (Sender oder Empfänger)", width=400)
        self.search_button = ft.Button("Suchen", on_click=self.search_user, icon=ft.Icons.SEARCH)
        self.search_results = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Step")),
                ft.DataColumn(ft.Text("Typ")),
                ft.DataColumn(ft.Text("Betrag")),
                ft.DataColumn(ft.Text("Sender")),
                ft.DataColumn(ft.Text("Empfänger")),
                ft.DataColumn(ft.Text("Zeitstempel")),
            ],
            rows=[]
        )
        self.view_search = ft.Container(
            content=ft.Column([
                ft.Row([self.search_input, self.search_button]),
                ft.Column([self.search_results], scroll=ft.ScrollMode.ADAPTIVE, expand=True)
            ], expand=True),
            key="search", expand=True
        )

        # --- VIEW 3: Dead Letter Queue (Failed Tasks) ---
        self.dlq_refresh_button = ft.Button("Fehler laden (Refresh)", on_click=self.load_dlq, icon=ft.Icons.REFRESH)
        self.dlq_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("ID")),
                ft.DataColumn(ft.Text("Zeit")),
                ft.DataColumn(ft.Text("Task")),
                ft.DataColumn(ft.Text("Event ID")),
                ft.DataColumn(ft.Text("Fehlermeldung")),
            ],
            rows=[]
        )
        self.view_dlq = ft.Container(
            content=ft.Column([
                ft.Row([self.dlq_refresh_button]),
                ft.Column([self.dlq_table], scroll=ft.ScrollMode.ADAPTIVE, expand=True)
            ], expand=True),
            key="dlq", expand=True
        )

        # --- Custom Animated Navigation ---
        self.tabs_row = ft.Row(spacing=10)
        self.tab_containers = []

        for i, (text, icon) in enumerate(
                [("Live Streaming", ft.Icons.MONITOR_HEART), ("User History", ft.Icons.PERSON_SEARCH),
                 ("Failed Tasks", ft.Icons.BUG_REPORT)]):
            tab = ft.Container(
                content=ft.Row([ft.Icon(icon), ft.Text(text, weight=ft.FontWeight.BOLD, size=16)]),
                padding=ft.Padding.symmetric(vertical=10, horizontal=20),
                border=ft.Border.only(bottom=ft.BorderSide(3, ft.Colors.TRANSPARENT)),
                ink=True,  # Erzeugt einen schönen Wellen-Effekt beim Klicken
                border_radius=ft.BorderRadius.only(top_left=5, top_right=5),
                on_click=lambda e, idx=i: self.switch_view(idx),
                animate=ft.Animation(300, ft.AnimationCurve.FAST_OUT_SLOWIN)  # Weicher Rahmen-Übergang
            )
            self.tab_containers.append(tab)
            self.tabs_row.controls.append(tab)

        # --- Animated Content Area ---
        self.content_area = ft.AnimatedSwitcher(
            content=self.view_monitor,
            transition=ft.AnimatedSwitcherTransition.FADE,
            duration=400,
            reverse_duration=400,
            switch_in_curve=ft.AnimationCurve.EASE_IN_OUT,
            switch_out_curve=ft.AnimationCurve.EASE_IN_OUT,
            expand=True
        )

        self.page.add(
            ft.Text("AML Fraud Detection - Dashboard", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400),
            self.tabs_row,
            ft.Divider(height=1, color=ft.Colors.WHITE24),
            self.content_area
        )
        self.switch_view(0)  # Initiiert das erste Styling

    def switch_view(self, idx: int):
        # 1. Update der Tab-Styles (Schriftfarbe und Unterstrich animieren)
        for i, tab in enumerate(self.tab_containers):
            if i == idx:
                tab.border = ft.Border.only(bottom=ft.BorderSide(3, ft.Colors.BLUE_400))
                tab.content.controls[0].color = ft.Colors.BLUE_400
                tab.content.controls[1].color = ft.Colors.BLUE_400
            else:
                tab.border = ft.Border.only(bottom=ft.BorderSide(3, ft.Colors.TRANSPARENT))
                tab.content.controls[0].color = ft.Colors.WHITE54
                tab.content.controls[1].color = ft.Colors.WHITE54

        # 2. Update des Inhalts (Löst die Fade-Animation aus)
        if idx == 0:
            self.content_area.content = self.view_monitor
        elif idx == 1:
            self.content_area.content = self.view_search
        elif idx == 2:
            self.content_area.content = self.view_dlq
            asyncio.create_task(self.load_dlq(None))

        self.page.update()

    async def search_user(self, e):
        user_id = self.search_input.value.strip()
        if not user_id:
            return

        self.search_button.disabled = True
        self.page.update()
        self.search_results.rows.clear()

        try:
            resp = await http_client.get(f"{API_URL}/user-history/{user_id}")
            if resp.status_code == 200:
                for r in resp.json():
                    row = ft.DataRow(cells=[
                        ft.DataCell(ft.Text(str(r["step"]))),
                        ft.DataCell(ft.Text(r["type"])),
                        ft.DataCell(ft.Text(f"{r['amount']:.2f}")),
                        ft.DataCell(ft.Text(r["nameOrig"])),
                        ft.DataCell(ft.Text(r["nameDest"])),
                        ft.DataCell(ft.Text(r["timestamp"][:19] if r["timestamp"] else "")),
                    ])
                    self.search_results.rows.append(row)
        except Exception as ex:
            print("Suchfehler:", ex)

        self.search_button.disabled = False
        self.page.update()

    async def clear_dashboard(self, e):
        self.clear_button.disabled = True
        self.page.update()
        try:
            resp = await http_client.delete(f"{API_URL}/transaction-state")
            if resp.status_code == 200:
                self.data_table.rows.clear()
                self.progress_text.value = "Dashboard und Datenbank-Status geleert."
            else:
                self.progress_text.value = f"Fehler beim Leeren: {resp.text}"
        except Exception as ex:
            self.progress_text.value = f"API Fehler: {ex}"

        self.clear_button.disabled = False
        self.page.update()

    async def load_dlq(self, e):
        self.dlq_refresh_button.disabled = True
        self.page.update()
        self.dlq_table.rows.clear()

        try:
            resp = await http_client.get(f"{API_URL}/dlq")
            if resp.status_code == 200:
                for r in resp.json():
                    error_text = str(r.get("error_msg", ""))
                    row = ft.DataRow(cells=[
                        ft.DataCell(ft.Text(str(r.get("id", "")))),
                        ft.DataCell(ft.Text(str(r.get("created_at", ""))[:19])),
                        ft.DataCell(ft.Text(str(r.get("task_name", "")))),
                        ft.DataCell(ft.Text(str(r.get("event_id", ""))[:8] + "...")),
                        ft.DataCell(ft.Text(error_text[:100] + ("..." if len(error_text) > 100 else ""))),
                    ])
                    self.dlq_table.rows.append(row)
        except Exception as ex:
            print("DLQ Ladefehler:", ex)

        self.dlq_refresh_button.disabled = False
        self.page.update()

    async def poll_status(self, event_id: str, row_cells: list):
        while True:
            try:
                async with self.poll_semaphore:
                    resp = await http_client.get(f"{API_URL}/transaction-status/{event_id}")
                    if resp.status_code == 200:
                        data = resp.json()
                        status = data["status"]
                        score = data["fraud_probability"]

                        row_cells[4].content.value = status
                        row_cells[5].content.value = f"{score:.4f}"

                        if status == "BLOCKED":
                            row_cells[4].content.color = ft.Colors.RED_400
                        elif status == "SUCCESS":
                            row_cells[4].content.color = ft.Colors.GREEN_400

                        self.page.update()

                        if status in ["SUCCESS", "BLOCKED"]:
                            break
            except Exception as e:
                print(f"Pooling error for event {event_id}: {e}")

            await asyncio.sleep(2)  # 2 Sekunden warten, entlastet die DB

    async def start_tests(self, e):
        self.start_button.disabled = True
        self.progress_text.value = f"Lese Daten aus {CSV_PATH} und schicke sie an API..."
        self.page.update()

        if not CSV_PATH.exists():
            self.progress_text.value = f"Fehler: CSV nicht gefunden ({CSV_PATH})"
            self.page.update()
            return

        with open(CSV_PATH, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                #if count >= 200:
                #    break

                payload = {
                    "amount": float(row["amount"]),
                    "nameOrig": row["nameOrig"],
                    "nameDest": row["nameDest"],
                    "type": row["type"],
                    "timestamp": datetime.now().isoformat()
                }

                for attempt in range(3):
                    try:
                        resp = await http_client.post(f"{API_URL}/transaction", json=payload)
                        if resp.status_code in (200, 202):
                            event_id = resp.json()["event_id"]

                            row_cells = [
                                ft.DataCell(ft.Text(event_id[:8] + "...")),
                                ft.DataCell(ft.Text(payload["nameOrig"])),
                                ft.DataCell(ft.Text(payload["nameDest"])),
                                ft.DataCell(ft.Text(f"{payload['amount']:.2f}")),
                                ft.DataCell(ft.Text("PENDING", color=ft.Colors.AMBER_400)),
                                ft.DataCell(ft.Text("...")),
                            ]
                            self.data_table.rows.insert(0, ft.DataRow(cells=row_cells))
                            self.page.update()

                            asyncio.create_task(self.poll_status(event_id, row_cells))
                            break
                    except Exception as ex:
                        if attempt == 2:
                            print("API Fehler:", ex)
                        await asyncio.sleep(0.5)

                count += 1
                await asyncio.sleep(0.02)

        self.progress_text.value = f"{count} Transaktionen gesendet. ML-Berechnungen laufen..."
        self.start_button.disabled = False
        self.page.update()


async def main(page: ft.Page):
    app = AMLDashboard(page)


if __name__ == "__main__":
    ft.run(main)
