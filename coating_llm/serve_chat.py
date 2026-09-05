# -*- coding: utf-8 -*-
"""涂釉 Web 对话服务：浏览器人工测试入口(仅本机使用)。"""
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT + "/coating_ai")

from model import CoatingLLM
from chat import chat_once
from data import mat_lib

mlib = mat_lib()
PORT = 8000
torch.set_num_threads(3)

tok = Tokenizer.from_file(os.path.join(HERE, "model", "tokenizer.json"))
model = CoatingLLM(vocab_size=10000, hidden=384, layers=6, heads=8, kv_heads=4,
                   intermediate=1024, max_seq=1024, dropout=0.0)
model.load_state_dict(torch.load(os.path.join(HERE, "out", "sft", "coating_best.pt"),
                                 map_location="cpu", weights_only=True))
model.eval()

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>涂釉 · 人工对话测试</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "PingFang SC", "Microsoft YaHei", sans-serif; background: #f4f5f7; height: 100vh; display: flex; flex-direction: column; }
  header { background: #1f2937; color: #fff; padding: 12px 20px; font-size: 16px; display: flex; justify-content: space-between; align-items: center; }
  header small { color: #9ca3af; margin-left: 12px; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px; }
  .msg { max-width: 78%; padding: 10px 14px; border-radius: 12px; line-height: 1.7; font-size: 14.5px; white-space: pre-wrap; word-break: break-word; }
  .user { align-self: flex-end; background: #3b82f6; color: #fff; border-bottom-right-radius: 4px; }
  .bot  { align-self: flex-start; background: #fff; color: #1f2937; border-bottom-left-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,.08); }
  .presets { padding: 6px 20px 2px; display: flex; flex-wrap: wrap; gap: 8px; }
  .presets button { border: 1px solid #cbd5e1; background: #fff; border-radius: 14px; padding: 4px 12px; font-size: 12.5px; color: #475569; cursor: pointer; }
  .presets button:hover { border-color: #3b82f6; color: #3b82f6; }
  .bar { display: flex; gap: 10px; padding: 12px 20px 16px; background: #eef0f3; align-items: center; }
  #q { flex: 1; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 10px; font-size: 14px; resize: none; height: 44px; outline: none; }
  #q:focus { border-color: #3b82f6; }
  #send { padding: 10px 22px; border: 0; border-radius: 10px; background: #3b82f6; color: #fff; font-size: 14px; cursor: pointer; }
  #send:disabled { background: #93c5fd; cursor: wait; }
  .temp { font-size: 12px; color: #64748b; display: flex; align-items: center; gap: 6px; white-space: nowrap; }
  .temp input { vertical-align: middle; }
</style>
</head>
<body>
<header>涂釉 · 人工对话测试<small>17.42M · 数字单token · 双锚校准区间</small><span id="state"></span></header>
<div id="chat"></div>
<div class="presets">
  <button onclick="fill(this)">请推断这个新配方的性能：环氧酚醛体系，IR190 60.0%、RF516 15.0%、正丁醇 20.0%、1510蜡 3.0%、RF401 2.0%，烘烤205°C×17min。</button>
  <button onclick="fill(this)">分析一下样本R01-01，它的性能是怎么来的？</button>
  <button onclick="fill(this)">对于环氧酚醛体系的配方，把固化剂比例提高约50%，性能会怎么变化？</button>
  <button onclick="fill(this)">如果把一个环氧酚醛配方中的溶剂比例提高约三分之一，会发生什么？</button>
  <button onclick="fill(this)">介绍一下原料RF516，有什么特性？</button>
  <button onclick="fill(this)">T弯总是偏大（偏脆），怎么改善？</button>
</div>
<div class="bar">
  <textarea id="q" placeholder="输入问题，回车发送…"></textarea>
  <span class="temp">温度(0=最稳)<input type="range" id="temp" min="0" max="100" value="0" style="width:90px"><span id="tempv">0.0</span></span>
  <button id="send" onclick="send()">发送</button>
</div>
<script>
const hist = [];
const chatEl = document.getElementById('chat');
const qEl = document.getElementById('q');
const sendBtn = document.getElementById('send');
const tempEl = document.getElementById('temp');
tempEl.oninput = () => document.getElementById('tempv').textContent = (tempEl.value / 100).toFixed(1);

function add(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
  return d;
}
function fill(b) { qEl.value = b.textContent; qEl.focus(); }

async function send() {
  const q = qEl.value.trim();
  if (!q || sendBtn.disabled) return;
  qEl.value = '';
  add('user', q);
  const holder = add('bot', '生成中…（CPU推理约需十几秒）');
  sendBtn.disabled = true;
  document.getElementById('state').textContent = '推理中…';
  try {
    const r = await fetch('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ q, history: hist, temperature: tempEl.value / 100 }) });
    const d = await r.json();
    if (d.error) { holder.textContent = '出错: ' + d.error; }
    else { holder.textContent = d.a; hist.push([q, d.a]); }
  } catch (e) { holder.textContent = '请求失败: ' + e; }
  sendBtn.disabled = false;
  document.getElementById('state').textContent = '';
  chatEl.scrollTop = chatEl.scrollHeight;
}
qEl.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
add('bot', '你好，我是涂釉，涂料领域的配方性能小专家。单轮问答模式：每个问题请写全上下文（体系、组成、烘烤、问题）。新配方推断中，组成占比由原料库实时计算校正，机理量、历史引用、区间和建议由模型生成。左侧按钮是一些常用问法。');
</script>
</body>
</html>"""


def ground_roles(q, ans):
    """把回答中的组成角色占比替换为按原料库实时计算的真值(确定性部分不由模型口算)。"""
    m = re.search(r"体系[，,]\s*(.+?)，\s*烘烤", q)
    if not m:
        return ans
    comp = {}
    for part in m.group(1).split("、"):
        mm = re.match(r"(.+?)\s*([\d.]+)\s*%", part.strip())
        if mm and mm.group(1).strip() in mlib:
            comp[mm.group(1).strip()] = float(mm.group(2))
    if not comp:
        return ans
    total = sum(comp.values())
    roles = {"树脂": 0.0, "固化剂": 0.0, "溶剂": 0.0, "助剂": 0.0, "颜料": 0.0}
    for k, v in comp.items():
        r = mlib[k].get("role")
        if r in roles:
            roles[r] += v / total * 100
    truth = ", ".join(f"{r}占{p:.1f}%" for r, p in roles.items() if p > 0.1)
    if "新配方：" not in ans:
        return ans
    head, rest = ans.split("新配方：", 1)
    for anchor in ("，酚羟基/环氧当量比", "，NCO/OH当量比", "，有效交联密度"):
        if anchor in rest:
            body, tail = rest.split(anchor, 1)
            return head + "新配方：" + truth + anchor + tail
    return ans


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            q = str(req.get("q", "")).strip()
            temp = min(max(float(req.get("temperature", 0.01)), 0.01), 1.0)
            if not q:
                self._send(400, json.dumps({"error": "问题为空"}).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            ans = chat_once(model, tok, q, history=(), temperature=temp,
                            top_p=0.9, top_k=10, repetition_penalty=1.05,
                            max_new_tokens=400)
            ans = ground_roles(q, ans)
            self._send(200, json.dumps({"a": ans}, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        except Exception as e:
            self._send(500, json.dumps({"error": repr(e)}, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")


if __name__ == "__main__":
    print(f"涂釉对话服务已启动: http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
