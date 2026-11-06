# -*- coding: utf-8 -*-
"""Acesso ao DSpace do ARCA: login, leitura, escrita (com travas).

LOGIN
  O ARCA autentica pela conta Microsoft da Fiocruz (OIDC). Nao ha login por
  senha na API. O script abre o Brave com a porta de depuracao (CDP), voce faz
  o login normalmente na janela, e o script le os cookies da sessao. O JWT
  fica no cookie `dsAuthInfo` e e enviado como `Authorization: Bearer`.
  Os cookies ficam em dados/sessao/ por 1 hora.

TRES TRAVAS DE ESCRITA (todas em `patch()`, o unico caminho de escrita):
  1. SOMENTE_LEITURA (config, opcional via ARCA_SOMENTE_LEITURA=sim):
     ligada, nenhuma escrita sai.
  2. dc.description.provenance: registro de workflow do DSpace. NUNCA e
     escrito, removido ou movido (um `remove` do campo inteiro ja apagou o
     historico do item icict/78513 em 2026-09-30, sem volta).
  3. remove de dcterms.provenance inteiro: proibido; o script so remove as
     linhas dele, por indice.
"""
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from urllib.parse import quote, unquote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

CAMPO_PROVENANCE_DSPACE = "dc.description.provenance"   # do DSpace: intocavel
CAMPO_PROVENANCE_SCRIPT = "dcterms.provenance"          # onde o script registra


class ErroDSpace(Exception):
    def __init__(self, msg, status=None, corpo=""):
        super().__init__(msg)
        self.status = status
        self.corpo = corpo


class ErroEscritaProibida(RuntimeError):
    """Tentativa de escrita bloqueada por uma das travas."""


# ---------------------------------------------------------------------------
# Travas
# ---------------------------------------------------------------------------
def conferir_operacoes(operacoes):
    """Levanta ErroEscritaProibida se alguma operacao violar as travas."""
    if config.SOMENTE_LEITURA:
        raise ErroEscritaProibida(
            "Modo SOMENTE-LEITURA ligado (arca/config.py): nada foi gravado no ARCA. "
            "Rode sem a variavel ARCA_SOMENTE_LEITURA=sim para gravar.")
    proibido = "/metadata/" + CAMPO_PROVENANCE_DSPACE
    inteiro_script = "/metadata/" + CAMPO_PROVENANCE_SCRIPT
    for op in operacoes or []:
        for chave in ("path", "from"):
            caminho = str(op.get(chave) or "")
            if caminho == proibido or caminho.startswith(proibido + "/"):
                raise ErroEscritaProibida(
                    "Operacao em %s bloqueada (%s %s=%s): e o registro de workflow "
                    "do DSpace." % (CAMPO_PROVENANCE_DSPACE, op.get("op"), chave, caminho))
        if op.get("op") == "remove" and op.get("path") == inteiro_script:
            raise ErroEscritaProibida(
                "remove de %s inteiro bloqueado: so as linhas do script podem sair."
                % CAMPO_PROVENANCE_SCRIPT)


# ---------------------------------------------------------------------------
# Sessao do navegador (CDP)
# ---------------------------------------------------------------------------
_trava = threading.Lock()


def _ler_cookies():
    try:
        with open(config.COOKIES_JSON, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("cookies") and time.time() < d.get("expira_em", 0):
            return d["cookies"]
    except Exception:
        pass
    return None


def _salvar_cookies(cookies):
    os.makedirs(os.path.dirname(config.COOKIES_JSON), exist_ok=True)
    with open(config.COOKIES_JSON, "w", encoding="utf-8") as f:
        json.dump({"cookies": cookies, "expira_em": time.time() + config.COOKIE_TTL_S}, f)


def esquecer_sessao():
    try:
        os.remove(config.COOKIES_JSON)
    except OSError:
        pass


def _brave():
    for base in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"),
                 os.environ.get("LOCALAPPDATA")):
        if base:
            p = os.path.join(base, "BraveSoftware", "Brave-Browser", "Application", "brave.exe")
            if os.path.isfile(p):
                return p
    return None


def _cdp_ativo():
    try:
        urllib.request.urlopen("http://localhost:%s/json/version" % config.CDP_PORTA, timeout=2)
        return True
    except Exception:
        return False


def login_no_navegador(aviso=print):
    """Abre o login Microsoft no Brave e devolve os cookies do ARCA."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    abriu = False
    if not _cdp_ativo():
        brave = _brave()
        if not brave:
            raise ErroDSpace("Brave nao encontrado. Instale o Brave ou abra um navegador "
                             "Chromium com --remote-debugging-port=%s." % config.CDP_PORTA)
        os.makedirs(config.PERFIL_NAVEGADOR, exist_ok=True)
        subprocess.Popen([brave, "--remote-debugging-port=%s" % config.CDP_PORTA,
                          "--user-data-dir=%s" % config.PERFIL_NAVEGADOR,
                          "--new-window", "--no-first-run", "--no-default-browser-check",
                          "about:blank"])
        for _ in range(20):
            time.sleep(1)
            if _cdp_ativo():
                abriu = True
                break
        else:
            raise ErroDSpace("O Brave nao abriu a porta de depuracao %s." % config.CDP_PORTA)

    opts = Options()
    opts.add_experimental_option("debuggerAddress", "localhost:%s" % config.CDP_PORTA)
    # o Selenium Manager escolhe o ChromeDriver pela versao do Chrome instalado, que pode estar a frente do Brave
    # ("ChromeDriver only supports Chrome version 156 / Current browser version is 155", 2026-10-20): pede o
    # driver da versao do navegador que esta mesmo na porta de depuracao
    try:
        versao = requests.get("http://localhost:%s/json/version" % config.CDP_PORTA, timeout=5).json()["Browser"]
        opts.browser_version = versao.split("/")[-1].split(".")[0]
        if _brave():
            opts.binary_location = _brave()
    except Exception:
        pass
    driver = webdriver.Chrome(service=Service(), options=opts)

    def cookies_do_arca():
        todos = driver.execute_cdp_cmd("Network.getAllCookies", {}).get("cookies", [])
        return {c["name"]: c["value"] for c in todos if config.HOSTNAME_UI in c.get("domain", "")}

    # o Brave ja pode estar logado (o responsavel logou noutra aba, ou o SSO renovou): usa esse login
    # antes de abrir outra aba de login (em 2026-10-20 as tentativas acumulavam abas sem perceber o login feito)
    try:
        cookies = cookies_do_arca()
        # a data do token nao basta: se o servidor reiniciou, o token antigo deixa de valer (2026-10-25)
        if _token_valido(cookies) and _token_aceito(cookies):
            _salvar_cookies(cookies)
            return cookies
    except Exception:
        pass
    url = ("https://login.microsoftonline.com/%s/oauth2/v2.0/authorize?client_id=%s"
           "&response_type=code&scope=openid+email+profile&redirect_uri=%s"
           % (config.OIDC_TENANT_ID, config.OIDC_CLIENT_ID,
              quote(config.OIDC_REDIRECT_URI, safe="")))
    aviso("Faca o login da Fiocruz na janela do Brave (ate %ds)..." % config.CDP_ESPERA_LOGIN_S)
    driver.get(url)

    # Espera o token, nao so a URL: a URL volta para o ARCA alguns segundos
    # antes de o Angular gravar dsAuthInfo (medido em 2026-10-02).
    cookies = {}
    for _ in range(config.CDP_ESPERA_LOGIN_S):
        time.sleep(1)
        try:
            host = urlparse(driver.current_url).hostname or ""
        except Exception:
            continue
        if config.HOSTNAME_UI not in host:
            continue
        cookies = cookies_do_arca()
        if _token(cookies) and _token_aceito(cookies):          # o token velho do Brave nao conta
            break
        # o site (Angular) fora do ar ou lento: a API ja deixou o cookie de autenticacao no retorno do OIDC;
        # troca direto por um token na propria API (2026-10-25: site com erro 502 e a API no ar)
        tok = _token_pela_api(cookies)
        if tok:
            cookies["dsAuthInfo"] = quote(json.dumps({"accessToken": tok}))
            break
    if not (_token(cookies) and _token_aceito(cookies)):
        raise ErroDSpace("Login nao concluido no tempo limite (sem token aceito pelo Arca; o site/API pode estar fora).")
    _salvar_cookies(cookies)
    if abriu:
        try:
            driver.execute_cdp_cmd("Browser.close", {})
        except Exception:
            pass
    return cookies


def _token_valido(cookies, folga_s=300):
    """Token presente e que ainda vale por pelo menos `folga_s` segundos (campo exp do JWT)."""
    import base64
    t = _token(cookies)
    if not t:
        return False
    try:
        corpo = t.split(".")[1]
        exp = json.loads(base64.urlsafe_b64decode(corpo + "=" * (-len(corpo) % 4))).get("exp", 0)
        return exp - time.time() > folga_s
    except Exception:
        return False


def _token_aceito(cookies):
    """O servidor aceita este token agora? (GET /authn/status com o Bearer)."""
    try:
        r = requests.get(config.BASE_URL + "/authn/status", timeout=config.HTTP_TIMEOUT,
                         headers={"Authorization": "Bearer " + (_token(cookies) or "")})
        return r.ok and bool(r.json().get("authenticated"))
    except Exception:
        return False


def _token_pela_api(cookies):
    """Com o cookie de autenticacao que o DSpace grava no retorno do OIDC ("Authorization-cookie"), pede o token
    JWT direto a API (POST /authn/login), como o Angular faria. None se ainda nao ha cookie ou se falhar."""
    if not cookies.get("Authorization-cookie"):
        return None
    try:
        s = requests.Session()
        for k, v in cookies.items():
            s.cookies.set(k, v, domain=config.DOMINIO_API)
        r = s.get(config.BASE_URL + "/authn/status", timeout=config.HTTP_TIMEOUT)
        xsrf = r.headers.get("DSPACE-XSRF-TOKEN") or cookies.get("DSPACE-XSRF-COOKIE")
        if xsrf:
            s.cookies.set("DSPACE-XSRF-COOKIE", xsrf, domain=config.DOMINIO_API)
        r = s.post(config.BASE_URL + "/authn/login", headers={"X-XSRF-TOKEN": xsrf or ""},
                   timeout=config.HTTP_TIMEOUT)
        auth = r.headers.get("Authorization", "")
        return auth.split(" ", 1)[1] if r.ok and auth.lower().startswith("bearer ") else None
    except Exception:
        return None


def _token(cookies):
    try:
        return json.loads(unquote(cookies.get("dsAuthInfo", ""))).get("accessToken")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cliente REST
# ---------------------------------------------------------------------------
class ClienteDSpace:
    def __init__(self, cookies, aviso=print):
        self.base = config.BASE_URL
        self.aviso = aviso
        self.s = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"], raise_on_status=False)
        ad = HTTPAdapter(pool_connections=config.HTTP_POOL, pool_maxsize=config.HTTP_POOL,
                         max_retries=retry)
        self.s.mount("https://", ad)
        self._usar_cookies(cookies)

    def _usar_cookies(self, cookies):
        self.s.cookies.clear()
        self.s.headers.clear()
        self.s.headers.update({"User-Agent": "arca-decs/9.0",
                               "Accept": "application/json, text/plain, */*"})
        for k, v in cookies.items():
            self.s.cookies.set(k, v, domain=config.DOMINIO_API)
        tok = _token(cookies)
        if tok:
            self.s.headers["Authorization"] = "Bearer " + tok
        csrf = cookies.get("DSPACE-XSRF-COOKIE") or cookies.get("XSRF-TOKEN")
        if csrf:
            self.s.headers["X-XSRF-TOKEN"] = csrf
        r = self.s.get(self.base + "/authn/status", timeout=config.HTTP_TIMEOUT)
        self._csrf(r)
        if not (r.ok and r.json().get("authenticated")):
            esquecer_sessao()
            raise ErroDSpace("Sessao do ARCA invalida ou expirada.")

    def _csrf(self, r):
        tok = r.headers.get("DSPACE-XSRF-TOKEN")
        if tok:
            self.s.headers["X-XSRF-TOKEN"] = tok
            self.s.cookies.set("DSPACE-XSRF-COOKIE", tok, domain=config.DOMINIO_API)

    def _pedir(self, metodo, url, **kw):
        kw.setdefault("timeout", config.HTTP_TIMEOUT)
        r = self.s.request(metodo, url, **kw)
        self._csrf(r)
        if r.status_code == 401:
            with _trava:
                self.aviso("Sessao expirou; refazendo o login...")
                esquecer_sessao()
                self._usar_cookies(login_no_navegador(self.aviso))
            r = self.s.request(metodo, url, **kw)
            self._csrf(r)
        return r

    def get(self, caminho, params=None, obrigatorio=True):
        url = caminho if caminho.startswith("http") else self.base + caminho
        r = self._pedir("GET", url, params=params)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 204 or not obrigatorio:
            return {}
        raise ErroDSpace("GET %s -> HTTP %s" % (url, r.status_code), r.status_code, r.text[:2000])

    def patch(self, caminho, operacoes):
        """Unico caminho de escrita do projeto. Passa pelas travas antes."""
        conferir_operacoes(operacoes)
        url = caminho if caminho.startswith("http") else self.base + caminho
        r = self._pedir("PATCH", url, json=operacoes,
                        headers={"Content-Type": "application/json-patch+json"})
        if r.status_code not in (200, 204):
            raise ErroDSpace("PATCH %s -> HTTP %s" % (url, r.status_code), r.status_code, r.text[:2000])
        return r.json() if r.content else {}

    def _escrita_politica(self, metodo, caminho, params=None, corpo=None):
        """Escrita de politicas de acesso (pedido do responsavel em 2026-10-16: restringir os bundles
        LICENSE). Mesma trava de SOMENTE-LEITURA do patch(); so aceita /authz/resourcepolicies."""
        if config.SOMENTE_LEITURA:
            raise ErroEscritaProibida("Modo SOMENTE-LEITURA ligado: nada foi gravado no ARCA.")
        if not caminho.startswith("/authz/resourcepolicies"):
            raise ErroEscritaProibida("Este metodo so escreve politicas de acesso (%s)." % caminho)
        url = self.base + caminho
        kw = {"params": params}
        if corpo is not None:
            kw["json"] = corpo
            kw["headers"] = {"Content-Type": "application/json"}
        r = self._pedir(metodo, url, **kw)
        if r.status_code not in (200, 201, 204):
            raise ErroDSpace("%s %s -> HTTP %s" % (metodo, url, r.status_code), r.status_code, r.text[:2000])
        return r.json() if r.content else {}

    def criar_politica(self, recurso_uuid, grupo_uuid, acao="READ", inicio=None, fim=None, tipo="TYPE_CUSTOM"):
        corpo = {"action": acao, "policyType": tipo}
        if inicio:
            corpo["startDate"] = inicio
        if fim:
            corpo["endDate"] = fim
        return self._escrita_politica("POST", "/authz/resourcepolicies",
                                      params={"resource": recurso_uuid, "group": grupo_uuid}, corpo=corpo)

    def apagar_politica(self, politica_id):
        return self._escrita_politica("DELETE", "/authz/resourcepolicies/%s" % politica_id)

    # -- bitstreams (substituicao de PDF por versao com OCR; autorizada pelo responsavel em 2026-10-19) --
    def adicionar_bitstream(self, bundle_uuid, nome, caminho):
        """Envia um arquivo novo ao bundle. So o ClienteDSpace escreve; respeita o modo SOMENTE-LEITURA."""
        if config.SOMENTE_LEITURA:
            raise ErroEscritaProibida("Modo SOMENTE-LEITURA ligado: nada foi gravado no ARCA.")
        with open(caminho, "rb") as f:
            conteudo = f.read()   # em memoria: se o login expirar, o pedido e repetido com o arquivo inteiro
        r = self._pedir("POST", self.base + "/core/bundles/%s/bitstreams" % bundle_uuid,
                        files={"file": (nome, conteudo, "application/pdf")})
        if r.status_code not in (200, 201):
            raise ErroDSpace("POST bitstreams -> HTTP %s" % r.status_code, r.status_code, r.text[:2000])
        return r.json()

    def apagar_bitstream(self, bitstream_uuid, bundle_uuid):
        """Remove um arquivo; so aceita bundle ORIGINAL (o LICENSE e os outros bundles ficam fora)."""
        if config.SOMENTE_LEITURA:
            raise ErroEscritaProibida("Modo SOMENTE-LEITURA ligado: nada foi gravado no ARCA.")
        if self.get("/core/bundles/%s" % bundle_uuid).get("name") != "ORIGINAL":
            raise ErroEscritaProibida("apagar_bitstream so remove arquivos do bundle ORIGINAL.")
        r = self._pedir("DELETE", self.base + "/core/bitstreams/%s" % bitstream_uuid)
        if r.status_code not in (200, 204):
            raise ErroDSpace("DELETE bitstream -> HTTP %s" % r.status_code, r.status_code, r.text[:2000])
        return True

    # -- estrutura de comunidades e colecoes (programas de pos-graduacao; pedido do responsavel em 2026-10-20) --
    _ESTRUTURA = ("/core/communities", "/core/collections/", "/core/collections?", "/eperson/groups/", "/core/items/")

    def _escrita_estrutura(self, metodo, caminho, **kw):
        """Cria comunidade/colecao/template/grupo de papel e move item. Mesma trava SOMENTE-LEITURA; so estes caminhos."""
        if config.SOMENTE_LEITURA:
            raise ErroEscritaProibida("Modo SOMENTE-LEITURA ligado: nada foi gravado no ARCA.")
        if not caminho.startswith(self._ESTRUTURA) or "/metadata/" + CAMPO_PROVENANCE_DSPACE in caminho:
            raise ErroEscritaProibida("Caminho de escrita de estrutura nao permitido: %s" % caminho)
        r = self._pedir(metodo, self.base + caminho, **kw)
        if r.status_code not in (200, 201, 204):
            raise ErroDSpace("%s %s -> HTTP %s" % (metodo, caminho, r.status_code), r.status_code, r.text[:2000])
        return r.json() if r.content else {}

    @staticmethod
    def _md(campos):
        return {k: [{"value": v, "language": None, "authority": None, "confidence": -1} for v in vs]
                for k, vs in campos.items() if vs}

    def criar_subcomunidade(self, pai_uuid, nome, campos):
        return self._escrita_estrutura("POST", "/core/communities?parent=%s" % pai_uuid,
                                       json={"name": nome, "metadata": self._md(dict(campos, **{"dc.title": [nome]}))})

    def criar_colecao(self, pai_uuid, nome, campos):
        return self._escrita_estrutura("POST", "/core/collections?parent=%s" % pai_uuid,
                                       json={"name": nome, "metadata": self._md(dict(campos, **{"dc.title": [nome]}))})

    def criar_template(self, colecao_uuid, campos):
        return self._escrita_estrutura("POST", "/core/collections/%s/itemtemplate" % colecao_uuid,
                                       json={"metadata": self._md(campos)})

    def criar_grupo_de_papel(self, colecao_uuid, papel):
        """papel: submittersGroup, adminGroup ou workflowGroups/<editor|reviewer|finaleditor>."""
        return self._escrita_estrutura("POST", "/core/collections/%s/%s" % (colecao_uuid, papel), json={"metadata": {}})

    def incluir_subgrupo(self, grupo_uuid, subgrupo_uuid):
        return self._escrita_estrutura("POST", "/eperson/groups/%s/subgroups" % grupo_uuid,
                                       data=self.base + "/eperson/groups/" + subgrupo_uuid,
                                       headers={"Content-Type": "text/uri-list"})

    def mover_item(self, item_uuid, colecao_uuid):
        return self._escrita_estrutura("PUT", "/core/items/%s/owningCollection" % item_uuid,
                                       data=self.base + "/core/collections/" + colecao_uuid,
                                       headers={"Content-Type": "text/uri-list"})

    def mapear_item(self, item_uuid, colecao_uuid):
        """Faz o item aparecer tambem em outra colecao (mapeamento); a colecao dona nao muda."""
        return self._escrita_estrutura("POST", "/core/items/%s/mappedCollections" % item_uuid,
                                       data=self.base + "/core/collections/" + colecao_uuid,
                                       headers={"Content-Type": "text/uri-list"})

    def desmapear_item(self, item_uuid, colecao_uuid):
        """Tira o item de uma colecao onde ele so aparece por mapeamento (a colecao dona nao muda; nada se apaga)."""
        dono = (self.get("/core/items/%s/owningCollection" % item_uuid) or {}).get("uuid")
        if dono == colecao_uuid:
            raise ErroEscritaProibida("a colecao %s e a dona do item: isso seria mover, nao desmapear" % colecao_uuid)
        return self._escrita_estrutura("DELETE", "/core/items/%s/mappedCollections/%s" % (item_uuid, colecao_uuid))

    def apagar_colecao_vazia(self, colecao_uuid):
        """So apaga colecao sem nenhum item (arquivado, retirado ou em submissao); confere antes."""
        c = self.get("/core/collections/%s" % colecao_uuid)
        if c.get("archivedItemsCount"):
            raise ErroEscritaProibida("colecao %s tem %s itens: nao apago" % (colecao_uuid, c.get("archivedItemsCount")))
        # itens arquivados/retirados e submissoes em andamento (revisao, supervisao, rascunho): apagar a colecao as perderia
        for conf in ("administrativeView", "workflowAdmin", "supervision", "workspace"):
            r = self.get("/discover/search/objects", params={"scope": colecao_uuid, "configuration": conf, "size": 1},
                         obrigatorio=False) or {}
            n = r.get("_embedded", {}).get("searchResult", {}).get("page", {}).get("totalElements", 0)
            if n:
                raise ErroEscritaProibida("colecao %s tem %s objeto(s) em '%s': nao apago" % (colecao_uuid, n, conf))
        return self._escrita_estrutura("DELETE", "/core/collections/%s" % colecao_uuid)

    # -- thumbnails (opcoes M; pedido do responsavel em 2026-10-16) --------------
    def requisicao_thumbnail(self, metodo, url, **kw):
        """Requisicoes do motor de thumbnails (arca/thumbnail_motor.py) com a sessao da suite.
        Sem trava propria (decisao do responsavel); so respeita o modo SOMENTE-LEITURA."""
        metodo = metodo.upper()
        if metodo not in ("GET", "HEAD", "OPTIONS") and config.SOMENTE_LEITURA:
            raise ErroEscritaProibida("Modo SOMENTE-LEITURA ligado: nada foi gravado no ARCA.")
        url = url if url.startswith("http") else self.base + url
        return self._pedir(metodo, url, **kw)

    # -- conveniencias ---------------------------------------------------------
    def item_por_handle(self, identificador):
        """Item a partir de handle, UUID, URL (/items/<uuid> ou /handle/<h>) ou hdl:."""
        tipo, valor = normalizar_identificador(identificador)
        if tipo == "uuid":
            return self.get("/core/items/" + valor, obrigatorio=False) or None
        if not valor:
            return None
        # /pid/find?id=<handle> e a rota que funciona nesta instancia
        # (/core/items/<handle> da 500 por causa da barra; medido em 2026-09-30)
        j = self.get("/pid/find", params={"id": valor}, obrigatorio=False)
        return j if j and j.get("uuid") else None


_RE_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def normalizar_identificador(texto):
    """('uuid', uuid) | ('handle', 'icict/12345') para qualquer forma colada.

    Aceita: UUID; https://arca.fiocruz.br/items/<uuid>; .../handle/icict/123;
    hdl:icict/123; icict/123; com espacos, tabulacao ou aspas em volta.
    """
    t = (texto or "").strip().strip('"').strip("'").strip()
    m = _RE_UUID.search(t)
    if m:
        return "uuid", m.group(0).lower()
    if "/handle/" in t:
        t = t.split("/handle/", 1)[1]
    if t.startswith("hdl:"):
        t = t[4:]
    return "handle", t.strip("/")


def conectar(aviso=print):
    """Cliente autenticado: reaproveita a sessao em cache ou abre o login."""
    cookies = _ler_cookies()
    if cookies:
        try:
            return ClienteDSpace(cookies, aviso)
        except ErroDSpace:
            pass
    return ClienteDSpace(login_no_navegador(aviso), aviso)
