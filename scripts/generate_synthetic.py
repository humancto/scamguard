#!/usr/bin/env python3
"""Generate provenance-tagged scam cases and hard negatives without live PII."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

GENERATOR_VERSION = 5
TARGETED_COUNTERFACTUAL_VERSION = 1

SYNTHETIC_REFERENCE_DEFAULT = (
    "https://consumer.ftc.gov/consumer-alerts/2025/03/what-are-signs-scam"
)

# These sources define scam mechanics and legitimate-channel safeguards. The generated messages
# below are original copy; source pages are never scraped or reproduced as training examples.
FAMILY_REFERENCE_URLS = {
    "benefit_identity_update": "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam",
    "jury_duty_penalty": "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam",
    "phantom_hacker_transfer": "https://www.ic3.gov/PSA/2023/PSA230929",
    "tax_refund_identity": "https://www.irs.gov/help/tax-scams",
    "task_unlock_deposit": "https://consumer.ftc.gov/consumer-alerts/2025/08/how-spot-avoid-task-scams",
    "bank_security_code": "https://consumer.ftc.gov/consumer-alerts/2024/03/never-move-your-money-protect-it-thats-scam",
    "family_bail_intermediary": "https://consumer.ftc.gov/articles/scammers-use-fake-emergencies-steal-your-money",
    "reshipping_job": "https://www.uspis.gov/wp-content/uploads/2021/08/uspis-be-smart-reshipping-scams-handout_508.pdf",
    "medicare_card_fee": "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam",
    "immigration_case_fee": "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam",
    "package_identity_form": "https://www.uspis.gov/news/scam-article/smishing-package-tracking-text-scams",
    "account_safe_harbor": "https://www.ic3.gov/PSA/2023/PSA230929",
    "benefit_portal_safe": "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam",
    "jury_duty_notice_safe": "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam",
    "bank_transfer_warning_safe": "https://consumer.ftc.gov/consumer-alerts/2024/03/never-move-your-money-protect-it-thats-scam",
    "tax_opt_in_safe": "https://www.irs.gov/privacy-disclosure/text-messages-from-the-irs",
    "task_job_safe": "https://consumer.ftc.gov/consumer-alerts/2025/08/how-spot-avoid-task-scams",
    "bank_code_warning_safe": "https://consumer.ftc.gov/consumer-alerts/2024/03/never-move-your-money-protect-it-thats-scam",
    "family_emergency_check_safe": "https://consumer.ftc.gov/articles/scammers-use-fake-emergencies-steal-your-money",
    "reshipping_warning_safe": "https://www.uspis.gov/wp-content/uploads/2021/08/uspis-be-smart-reshipping-scams-handout_508.pdf",
    "medicare_card_safe": "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam",
    "immigration_case_safe": "https://consumer.ftc.gov/articles/how-avoid-government-impersonation-scam",
    "package_tracking_opt_in_safe": "https://www.uspis.gov/news/scam-article/smishing-package-tracking-text-scams",
    "remote_support_booking_safe": "https://www.ic3.gov/PSA/2023/PSA230929",
}

SCAM_FAMILIES = {
    "delivery_toll": (
        "DELIVERY_TOLL",
        "Unpaid {road} toll of {amount}. Pay now at {url} to avoid a {penalty} fee.",
    ),
    "delivery_parcel": (
        "DELIVERY_TOLL",
        "Your {carrier} parcel is on hold. Confirm your address and {amount} redelivery fee: {url}",
    ),
    "bank_otp": (
        "CREDENTIAL_THEFT",
        "{bank} fraud desk: reply with the {digits}-digit code we just sent to stop a {amount} charge.",
    ),
    "bank_link": (
        "CREDENTIAL_THEFT",
        "{bank}: unusual sign-in detected. Verify your password immediately at {url}.",
    ),
    "invoice_change": (
        "FINANCIAL",
        "The account for invoice {invoice} changed. Send {amount} today to {payment}; keep this confidential.",
    ),
    "advance_fee": (
        "FINANCIAL",
        "Your {amount} grant is approved. Pay a {fee} processing fee by {payment} before release.",
    ),
    "investment_guarantee": (
        "FINANCIAL",
        "Guaranteed {return_rate}% weekly returns with zero risk. Fund your account using {payment}: {url}",
    ),
    "job_check": (
        "OPPORTUNITY",
        "You are hired as a remote {role}. Deposit our check, then buy equipment from {url} today.",
    ),
    "job_fee": (
        "OPPORTUNITY",
        "Earn {amount} weekly from home. A refundable {fee} onboarding fee reserves your position.",
    ),
    "romance_emergency": (
        "RELATIONSHIP",
        "I cannot call from the hospital, love. Please send {amount} in {payment} and do not tell anyone.",
    ),
    "authority_threat": (
        "IDENTITY_IMPERSONATION",
        "This is Agent {surname}. A warrant is pending. Stay on the line and pay {amount} by {payment}.",
    ),
    "tech_support": (
        "CREDENTIAL_THEFT",
        "{vendor} security detected a virus. Install remote support from {url} and share the session code.",
    ),
    "marketplace_offplatform": (
        "OPPORTUNITY",
        "I can pay {amount} for the item. {marketplace} says you must confirm the seller card at {url}; send me the confirmation screenshot.",
    ),
    "vendor_bank_change": (
        "FINANCIAL",
        "Hi, this is {vendor} accounts. For invoice {invoice}, our settlement details changed; use {payment} for the next payment and reply when processed.",
    ),
    "family_new_number": (
        "IDENTITY_IMPERSONATION",
        "Hi {relative}, my phone broke and this is my temporary number. Could you send {amount} by {payment}? I will explain after work.",
    ),
    "recruiter_identity": (
        "OPPORTUNITY",
        "{vendor} would like to interview you for the {role} role. Complete the background check by uploading ID and bank details at {url}.",
    ),
    "refund_recovery": (
        "FINANCIAL",
        "We traced your lost {amount}. Release requires a refundable {fee} recovery bond paid through {payment}.",
    ),
    "quiet_account_takeover": (
        "CREDENTIAL_THEFT",
        "{vendor} moved customer accounts to a new portal. Sign in at {url} before {day} to retain access.",
    ),
    "executive_payment": (
        "IDENTITY_IMPERSONATION",
        "I am in a confidential board meeting. Process {amount} to our new {vendor} account by {payment}; I will approve the paperwork later.",
    ),
    "relationship_travel": (
        "RELATIONSHIP",
        "I have been planning our future together. My return ticket is blocked until I pay {amount}; please use {payment} so I can finally visit.",
    ),
    "credential_shared_document": (
        "CREDENTIAL_THEFT",
        "{vendor} shared the revised contract with you. Your mailbox session expired; sign in at {url} to view it before {day}.",
    ),
    "identity_case_callback": (
        "IDENTITY_IMPERSONATION",
        "Detective {surname} needs to verify your identity for case {invoice}. Do not call the station; send your ID and banking details here.",
    ),
    "delivery_parking": (
        "DELIVERY_TOLL",
        "Parking notice {invoice}: settle {amount} at {url} before {day} to prevent vehicle registration restrictions.",
    ),
    "financial_payroll_change": (
        "FINANCIAL",
        "Payroll migration for {vendor}: confirm your salary account and routing details at {url} before the next pay cycle.",
    ),
    "relationship_inheritance": (
        "RELATIONSHIP",
        "After everything we have shared, I trust only you. A family inheritance is held for a {fee} clearance bond; please send it by {payment}.",
    ),
    "benefit_identity_update": (
        "IDENTITY_IMPERSONATION",
        "{agency} benefits desk: your {benefit} is suspended. Send your identity number and bank details at {url} before {day}.",
    ),
    "jury_duty_penalty": (
        "IDENTITY_IMPERSONATION",
        "Court officer {surname}: you missed jury duty under case {invoice}. Pay {amount} by {payment} now to cancel the arrest order.",
    ),
    "phantom_hacker_transfer": (
        "FINANCIAL",
        "{bank} security says a hacker reached your savings. Move {amount} to our protected {vendor} account by {payment} and do not tell anyone.",
    ),
    "tax_refund_identity": (
        "IDENTITY_IMPERSONATION",
        "{agency} refund unit approved {amount}. Confirm your tax identity and deposit account at {url} before {day}.",
    ),
    "task_unlock_deposit": (
        "OPPORTUNITY",
        "Your {role} tasks earned {amount}. Deposit {fee} by {payment} to unlock the next set and withdraw your commission.",
    ),
    "bank_security_code": (
        "CREDENTIAL_THEFT",
        "{bank} fraud team: a {amount} transfer is pending. Reply with the {digits}-digit verification code so we can block it.",
    ),
    "family_bail_intermediary": (
        "IDENTITY_IMPERSONATION",
        "I represent your {relative}, who was detained after an accident. Keep this private and send {amount} by {payment} for bail today.",
    ),
    "reshipping_job": (
        "OPPORTUNITY",
        "{vendor} hired you as a package inspector for {amount} weekly. Pay the {fee} activation charge at {url} before shipments begin.",
    ),
    "medicare_card_fee": (
        "IDENTITY_IMPERSONATION",
        "{agency} card services: pay {fee} at {url} and enter your benefit number to receive the required replacement card by {day}.",
    ),
    "immigration_case_fee": (
        "IDENTITY_IMPERSONATION",
        "{agency} case {invoice} has a compliance hold. Send {amount} by {payment} before {day} or your application will be cancelled.",
    ),
    "package_identity_form": (
        "DELIVERY_TOLL",
        "{carrier} could not release parcel {invoice}. Enter your address, card, and identity details at {url} and pay {fee}.",
    ),
    "account_safe_harbor": (
        "FINANCIAL",
        "A foreign attacker accessed your {bank} profile. Transfer {amount} by {payment} to the {agency} safe-harbor account for protection.",
    ),
}

HARD_NEGATIVE_FAMILIES = {
    "otp_warning": (
        "SAFE",
        "Your {bank} verification code is {digits}. Never share this code; employees will never ask for it.",
    ),
    "bank_notice": (
        "SAFE",
        "{bank}: A {amount} card purchase was approved {day}. If this was not you, call the number on your card.",
    ),
    "appointment": (
        "SAFE",
        "Reminder: your {clinic} appointment is {day} at {time}. Reply C to confirm or call our office.",
    ),
    "known_invoice": (
        "SAFE",
        "Invoice {invoice} from {vendor} is attached for the work completed. Payment details are unchanged.",
    ),
    "family_request": (
        "SAFE",
        "Could you pick up a {gift_card} gift card {day}? It is for {occasion} tonight.",
    ),
    "security_training": (
        "SAFE",
        "{vendor} {day} security training: urgent requests for passwords, crypto, or gift cards may be scams. Report them.",
    ),
    "promotion": (
        "UNCERTAIN",
        "{store}: {discount}% off ends tonight. Shop the sale at {url}. Reply STOP to unsubscribe.",
    ),
    "donation": (
        "UNCERTAIN",
        "{vendor}: Please support our local {cause} fundraiser. Learn about the registered charity at {url}.",
    ),
    "official_app_alert": (
        "SAFE",
        "{bank}: We declined a {amount} transaction {day}. Open the official app or call the number on your card; we will never ask for a code.",
    ),
    "recruiter_schedule": (
        "SAFE",
        "{vendor} scheduled your {role} interview for {day}. No payment or banking information is required before the interview.",
    ),
    "marketplace_platform": (
        "SAFE",
        "{marketplace}: Your {amount} sale is pending {day}. Keep payment in the app and never share verification codes with a buyer.",
    ),
    "family_transfer_verified": (
        "SAFE",
        "Hi {relative}, could we discuss the {amount} transfer {day}? Please call me on the number you already have before sending anything.",
    ),
    "vendor_change_verified": (
        "SAFE",
        "{vendor}: A bank-detail change for invoice {invoice} is pending dual approval {day}. Do not use it until your known contact confirms by phone.",
    ),
    "delivery_pickup": (
        "SAFE",
        "Your {carrier} parcel is ready for pickup {day} at {time}. Bring photo ID; no online payment is due.",
    ),
    "executive_workflow": (
        "SAFE",
        "{vendor} payment request {invoice} is in the approved finance queue. Use the recorded supplier details; no confidential transfer is requested.",
    ),
    "partner_travel_verified": (
        "SAFE",
        "Hi {relative}, my return trip is booked for {day} at {time}. I do not need money; we can review the itinerary together on our usual video call.",
    ),
    "shared_document_safe": (
        "SAFE",
        "{vendor} shared contract {invoice} in the official workspace. Open the saved company app; this message does not ask for a password.",
    ),
    "police_callback_safe": (
        "SAFE",
        "{road} community notice: verify any call claiming to be Detective {surname} by using the station number on the official city website. Never send ID in chat.",
    ),
    "parking_app_notice": (
        "SAFE",
        "Parking session {invoice} ends {day}. If you need more time, open the city parking app you already installed; no payment link is included.",
    ),
    "payroll_change_safe": (
        "SAFE",
        "{vendor} payroll reminder {invoice}: direct-deposit changes before {day} require an in-person ID check. No bank details should be sent by email or chat.",
    ),
    "relationship_known_contact": (
        "SAFE",
        "Hi {relative}, I am home safely {day} and do not need a {amount} transfer. Call me at the number you already know if any message claims I need emergency money.",
    ),
    "benefit_portal_safe": (
        "SAFE",
        "{agency} information notice for {benefit}: no payment or bank details are requested. Check your saved official account after {day}.",
    ),
    "jury_duty_notice_safe": (
        "SAFE",
        "Court reminder for case {invoice}: jury questions must be verified with the clerk using the official directory. Do not pay by {payment}.",
    ),
    "bank_transfer_warning_safe": (
        "SAFE",
        "{bank} security reminder for {day}: we will never tell you to move {amount} to a protected account. Call the number on your card.",
    ),
    "tax_opt_in_safe": (
        "SAFE",
        "Your opted-in {agency} appointment reminder is for {day} at {time}. This text does not request payment, identity, or bank information.",
    ),
    "task_job_safe": (
        "SAFE",
        "{vendor} interview for the {role} role is {day} at {time}. You will never deposit {fee} or pay to unlock work.",
    ),
    "bank_code_warning_safe": (
        "SAFE",
        "Your {bank} {digits}-digit verification code confirms a {amount} action. Never give this code to a caller; contact the bank directly.",
    ),
    "family_emergency_check_safe": (
        "SAFE",
        "If anyone claims {relative} needs {amount} for bail, pause and call the family number you already know. Do not use {payment}.",
    ),
    "reshipping_warning_safe": (
        "SAFE",
        "{vendor} hiring notice: legitimate package roles never require a {fee} activation payment or reshipping goods bought by strangers.",
    ),
    "medicare_card_safe": (
        "SAFE",
        "{agency} safety notice: replacement benefit cards are free. Do not pay {fee} or enter your identity number at an unexpected link.",
    ),
    "immigration_case_safe": (
        "SAFE",
        "For case {invoice}, use the saved {agency} portal to review notices. Officials will not demand {amount} by {payment} over chat.",
    ),
    "package_tracking_opt_in_safe": (
        "SAFE",
        "Your opted-in {carrier} update for parcel {invoice} is available in the official app. No {fee} payment or identity form is required.",
    ),
    "remote_support_booking_safe": (
        "SAFE",
        "Your scheduled {vendor} support appointment is {day} at {time}. The technician will not ask you to move {amount} or hide the call.",
    ),
}

# Schema-v12 train-only correction families. They are independently worded from regression rows
# and target behavior categories exposed by the schema-v11 error audit, not individual examples.
TARGETED_COUNTERFACTUAL_FAMILIES = {
    "known_channel_transfer_safe": (
        "SAFE",
        "NONE",
        "Before sending {amount} for {occasion}, let us confirm the details on our regular call. I will use your saved contact and wait for approval.",
        "https://consumer.ftc.gov/articles/scammers-use-fake-emergencies-steal-your-money",
    ),
    "new_number_transfer_scam": (
        "SCAM",
        "IDENTITY_IMPERSONATION",
        "It is me, {relative}, on a borrowed phone. Send {amount} through {payment} now and do not ring my usual number because I cannot answer.",
        "https://consumer.ftc.gov/articles/scammers-use-fake-emergencies-steal-your-money",
    ),
    "family_status_safe": (
        "SAFE",
        "NONE",
        "Hi {relative}, I arrived safely {day}. I am not requesting money. Ignore anyone using my name for an emergency and reach me through our normal contact.",
        "https://consumer.ftc.gov/articles/scammers-use-fake-emergencies-steal-your-money",
    ),
    "private_emergency_payment_scam": (
        "SCAM",
        "RELATIONSHIP",
        "{relative}, I am in an emergency and cannot speak. Keep this private and send {amount} by {payment} before {time}; please do not verify with anyone.",
        "https://consumer.ftc.gov/articles/scammers-use-fake-emergencies-steal-your-money",
    ),
    "marketplace_in_app_safe": (
        "SAFE",
        "NONE",
        "{marketplace} order {invoice} is paid inside the app. Ship only when your saved app marks it ready; no buyer needs a code, refund, or outside payment.",
        "https://consumer.ftc.gov/articles/how-spot-avoid-and-report-fake-check-scams",
    ),
    "marketplace_refund_scam": (
        "SCAM",
        "OPPORTUNITY",
        "I accidentally paid {amount} extra for listing {invoice}. Send the difference back using {payment} today before {marketplace} cancels the sale.",
        "https://consumer.ftc.gov/articles/how-spot-avoid-and-report-fake-check-scams",
    ),
    "official_app_review_safe": (
        "SAFE",
        "NONE",
        "{bank}: A transfer review is available {day}. Open the app from your home screen or use the number on your card; this message has no link and asks for no credentials.",
        "https://consumer.ftc.gov/consumer-alerts/2024/03/never-move-your-money-protect-it-thats-scam",
    ),
    "bank_review_link_scam": (
        "SCAM",
        "CREDENTIAL_THEFT",
        "{bank}: A {amount} transfer is blocked. Sign in at {url} and enter the {digits}-digit code now or the account will be restricted.",
        "https://consumer.ftc.gov/consumer-alerts/2024/03/never-move-your-money-protect-it-thats-scam",
    ),
}

VALUES = {
    "road": ["Bay Area", "Central", "Metro", "North County"],
    "amount": ["$8.60", "$47", "$325", "$1,240"],
    "penalty": ["$35", "$50", "$75"],
    "url": ["https://verify.example/a7", "http://account.example/z9", "verify[.]example/help"],
    "carrier": ["postal", "express", "courier", "shipping"],
    "bank": ["Harbor Bank", "Union Credit", "Community Bank", "First Metro"],
    "digits": ["4", "6", "8"],
    "invoice": ["1048", "A-774", "Q3-219"],
    "payment": ["gift cards", "cryptocurrency", "a wire transfer", "a payment app"],
    "fee": ["$29", "$75", "$199"],
    "return_rate": ["18", "25", "40"],
    "role": ["assistant", "mystery shopper", "data clerk", "reviewer"],
    "surname": ["Miller", "Patel", "Garcia", "Wilson"],
    "vendor": ["Northstar", "Contoso", "Citywide", "Pioneer"],
    "clinic": ["dental", "vision", "family medicine", "physical therapy"],
    "day": ["Monday", "Thursday", "tomorrow", "August 28"],
    "time": ["9:30 AM", "2 PM", "4:15 PM"],
    "gift_card": ["grocery", "restaurant", "bookstore"],
    "occasion": ["the birthday", "teacher appreciation", "the team dinner"],
    "store": ["Corner Market", "Northstar Books", "City Outfitters"],
    "discount": ["15", "25", "40"],
    "cause": ["food bank", "animal shelter", "school"],
    "relative": ["Mum", "Dad", "Auntie", "Grandpa"],
    "marketplace": ["LocalList", "TownMarket", "ResaleHub", "Community Shop"],
    "agency": ["Benefits Office", "Revenue Service", "Health Benefits Agency", "Civic Services"],
    "benefit": ["retirement payment", "health coverage", "tax credit", "family allowance"],
}

# Curated translations of safety-critical lookalikes. These are not
# machine-translated copies of evaluation rows: they are synthetic templates
# whose family stays in one split in every language. Brand-like names, amounts,
# invoice IDs, and one-time-code lengths are intentionally shared variables.
# Native-speaker review remains a production-release gate.
MULTILINGUAL_SAFE_TEMPLATES = {
    "Spanish": {
        "otp_warning": "Tu código de verificación de {bank} es {digits}. No compartas nunca este código; el personal nunca te lo pedirá.",
        "appointment": "Recordatorio: tu cita de {clinic} es el {day} a las {time}. Responde C para confirmar o llama al consultorio.",
        "known_invoice": "Se adjunta la factura {invoice} de {vendor} por el trabajo realizado. Los datos de pago no han cambiado.",
        "security_training": "Formación de seguridad de {vendor} el {day}: las solicitudes urgentes de contraseñas, criptomonedas o tarjetas regalo pueden ser estafas. Repórtalas.",
        "executive_workflow": "La solicitud de pago {invoice} de {vendor} está en la cola financiera aprobada. Usa los datos registrados del proveedor; no se solicita ninguna transferencia confidencial.",
        "bank_notice": "{bank}: se aprobó una compra con tarjeta de {amount} el {day}. Si no fuiste tú, llama al número que figura en tu tarjeta.",
        "recruiter_schedule": "{vendor} programó tu entrevista para el puesto de {role} el {day}. No se requiere ningún pago ni información bancaria antes de la entrevista.",
        "delivery_pickup": "Tu paquete de {carrier} está listo para recoger el {day} a las {time}. Lleva una identificación; no hay ningún pago en línea pendiente.",
        "shared_document_safe": "{vendor} compartió el contrato {invoice} en el espacio de trabajo oficial. Abre la aplicación guardada de la empresa; este mensaje no pide tu contraseña.",
        "official_app_alert": "{bank}: rechazamos una transacción de {amount} el {day}. Abre la aplicación oficial o llama al número de tu tarjeta; nunca te pediremos un código.",
        "parking_app_notice": "La sesión de estacionamiento {invoice} termina el {day}. Si necesitas más tiempo, abre la aplicación municipal que ya tienes instalada; no se incluye ningún enlace de pago.",
        "payroll_change_safe": "Recordatorio de nómina {invoice} de {vendor}: los cambios de depósito directo antes del {day} requieren verificar la identidad en persona. No envíes datos bancarios por correo o chat.",
    },
    "Dutch": {
        "otp_warning": "Uw verificatiecode van {bank} is {digits}. Deel deze code nooit; medewerkers zullen er nooit om vragen.",
        "appointment": "Herinnering: uw afspraak bij {clinic} is {day} om {time}. Antwoord C om te bevestigen of bel de praktijk.",
        "known_invoice": "Factuur {invoice} van {vendor} voor het uitgevoerde werk is bijgevoegd. De betaalgegevens zijn niet gewijzigd.",
        "security_training": "Beveiligingstraining van {vendor} op {day}: dringende verzoeken om wachtwoorden, cryptovaluta of cadeaukaarten kunnen oplichting zijn. Meld ze.",
        "executive_workflow": "Betaalverzoek {invoice} van {vendor} staat in de goedgekeurde financiële wachtrij. Gebruik de geregistreerde leveranciersgegevens; er wordt geen vertrouwelijke overboeking gevraagd.",
        "bank_notice": "{bank}: op {day} is een kaartbetaling van {amount} goedgekeurd. Was u dit niet, bel dan het nummer op uw kaart.",
        "recruiter_schedule": "{vendor} heeft uw gesprek voor de functie {role} gepland op {day}. Voor het gesprek zijn geen betaling of bankgegevens nodig.",
        "delivery_pickup": "Uw pakket van {carrier} kan op {day} om {time} worden opgehaald. Neem een identiteitsbewijs mee; er is geen online betaling verschuldigd.",
        "shared_document_safe": "{vendor} heeft contract {invoice} gedeeld in de officiële werkomgeving. Open de opgeslagen bedrijfsapp; dit bericht vraagt niet om een wachtwoord.",
        "official_app_alert": "{bank}: we hebben op {day} een transactie van {amount} geweigerd. Open de officiële app of bel het nummer op uw kaart; we vragen nooit om een code.",
        "parking_app_notice": "Parkeersessie {invoice} eindigt op {day}. Open zo nodig de gemeentelijke parkeerapp die u al hebt geïnstalleerd; dit bericht bevat geen betaallink.",
        "payroll_change_safe": "Salarisherinnering {invoice} van {vendor}: wijzigingen in de betaalrekening vóór {day} vereisen identiteitscontrole ter plaatse. Stuur geen bankgegevens via e-mail of chat.",
    },
    "French": {
        "otp_warning": "Votre code de vérification {bank} est {digits}. Ne communiquez jamais ce code ; aucun employé ne vous le demandera.",
        "appointment": "Rappel : votre rendez-vous de {clinic} est prévu {day} à {time}. Répondez C pour confirmer ou appelez le cabinet.",
        "known_invoice": "La facture {invoice} de {vendor} pour le travail effectué est jointe. Les coordonnées de paiement n'ont pas changé.",
        "security_training": "Formation de sécurité {vendor} le {day} : les demandes urgentes de mots de passe, de cryptomonnaie ou de cartes-cadeaux peuvent être frauduleuses. Signalez-les.",
        "executive_workflow": "La demande de paiement {invoice} de {vendor} figure dans la file financière approuvée. Utilisez les coordonnées fournisseur enregistrées ; aucun virement confidentiel n'est demandé.",
        "bank_notice": "{bank} : un achat par carte de {amount} a été approuvé {day}. Si vous n'en êtes pas à l'origine, appelez le numéro figurant sur votre carte.",
        "recruiter_schedule": "{vendor} a programmé votre entretien pour le poste de {role} {day}. Aucun paiement ni renseignement bancaire n'est requis avant l'entretien.",
        "delivery_pickup": "Votre colis {carrier} sera prêt à être retiré {day} à {time}. Munissez-vous d'une pièce d'identité ; aucun paiement en ligne n'est dû.",
        "shared_document_safe": "{vendor} a partagé le contrat {invoice} dans l'espace de travail officiel. Ouvrez l'application d'entreprise déjà enregistrée ; ce message ne demande pas de mot de passe.",
        "official_app_alert": "{bank} : nous avons refusé une transaction de {amount} {day}. Ouvrez l'application officielle ou appelez le numéro sur votre carte ; nous ne demanderons jamais de code.",
        "parking_app_notice": "La session de stationnement {invoice} se termine {day}. Pour la prolonger, ouvrez l'application municipale déjà installée ; ce message ne contient aucun lien de paiement.",
        "payroll_change_safe": "Rappel de paie {invoice} de {vendor} : toute modification du compte de versement avant {day} exige un contrôle d'identité en personne. N'envoyez aucune donnée bancaire par e-mail ou chat.",
    },
    "German": {
        "otp_warning": "Ihr Bestätigungscode von {bank} lautet {digits}. Geben Sie diesen Code niemals weiter; Mitarbeitende werden Sie nie danach fragen.",
        "appointment": "Erinnerung: Ihr Termin bei {clinic} ist am {day} um {time}. Antworten Sie zur Bestätigung mit C oder rufen Sie die Praxis an.",
        "known_invoice": "Rechnung {invoice} von {vendor} für die ausgeführte Arbeit ist beigefügt. Die Zahlungsdaten sind unverändert.",
        "security_training": "Sicherheitsschulung von {vendor} am {day}: Dringende Anfragen nach Passwörtern, Kryptowährung oder Geschenkkarten können Betrug sein. Melden Sie sie.",
        "executive_workflow": "Die Zahlungsanforderung {invoice} von {vendor} befindet sich in der genehmigten Finanzwarteschlange. Verwenden Sie die hinterlegten Lieferantendaten; es wird keine vertrauliche Überweisung verlangt.",
        "bank_notice": "{bank}: Am {day} wurde ein Kartenkauf über {amount} genehmigt. Wenn Sie das nicht waren, rufen Sie die Nummer auf Ihrer Karte an.",
        "recruiter_schedule": "{vendor} hat Ihr Gespräch für die Stelle {role} am {day} angesetzt. Vor dem Gespräch sind weder eine Zahlung noch Bankdaten erforderlich.",
        "delivery_pickup": "Ihr Paket von {carrier} kann am {day} um {time} abgeholt werden. Bringen Sie einen Ausweis mit; es ist keine Online-Zahlung fällig.",
        "shared_document_safe": "{vendor} hat Vertrag {invoice} im offiziellen Arbeitsbereich freigegeben. Öffnen Sie die bereits gespeicherte Firmen-App; diese Nachricht fragt nicht nach einem Passwort.",
        "official_app_alert": "{bank}: Wir haben am {day} eine Transaktion über {amount} abgelehnt. Öffnen Sie die offizielle App oder rufen Sie die Nummer auf Ihrer Karte an; wir fragen nie nach einem Code.",
        "parking_app_notice": "Die Parksitzung {invoice} endet am {day}. Öffnen Sie bei Bedarf die bereits installierte städtische Park-App; diese Nachricht enthält keinen Zahlungslink.",
        "payroll_change_safe": "Gehaltsabrechnungshinweis {invoice} von {vendor}: Änderungen der Bankverbindung vor {day} erfordern eine persönliche Identitätsprüfung. Senden Sie keine Bankdaten per E-Mail oder Chat.",
    },
    "Italian": {
        "otp_warning": "Il tuo codice di verifica {bank} è {digits}. Non condividerlo mai; nessun dipendente te lo chiederà.",
        "appointment": "Promemoria: l'appuntamento di {clinic} è {day} alle {time}. Rispondi C per confermare o chiama lo studio.",
        "known_invoice": "È allegata la fattura {invoice} di {vendor} per il lavoro svolto. I dati di pagamento non sono cambiati.",
        "security_training": "Formazione sulla sicurezza di {vendor} il {day}: richieste urgenti di password, criptovalute o buoni regalo possono essere truffe. Segnalale.",
        "executive_workflow": "La richiesta di pagamento {invoice} di {vendor} è nella coda finanziaria approvata. Usa i dati del fornitore registrati; non è richiesto alcun bonifico riservato.",
        "bank_notice": "{bank}: un acquisto con carta di {amount} è stato approvato il {day}. Se non eri tu, chiama il numero riportato sulla carta.",
        "recruiter_schedule": "{vendor} ha fissato il colloquio per il ruolo di {role} il {day}. Prima del colloquio non sono richiesti pagamenti né dati bancari.",
        "delivery_pickup": "Il pacco di {carrier} è pronto per il ritiro il {day} alle {time}. Porta un documento; non è dovuto alcun pagamento online.",
        "shared_document_safe": "{vendor} ha condiviso il contratto {invoice} nell'area di lavoro ufficiale. Apri l'app aziendale già salvata; questo messaggio non chiede una password.",
        "official_app_alert": "{bank}: abbiamo rifiutato una transazione di {amount} il {day}. Apri l'app ufficiale o chiama il numero sulla carta; non chiederemo mai un codice.",
        "parking_app_notice": "La sessione di parcheggio {invoice} termina il {day}. Se serve più tempo, apri l'app comunale già installata; il messaggio non include link di pagamento.",
        "payroll_change_safe": "Promemoria paghe {invoice} di {vendor}: le modifiche all'accredito prima del {day} richiedono una verifica di persona. Non inviare dati bancari via e-mail o chat.",
    },
    "Indonesian": {
        "otp_warning": "Kode verifikasi {bank} Anda adalah {digits}. Jangan pernah membagikan kode ini; petugas tidak akan pernah memintanya.",
        "appointment": "Pengingat: janji {clinic} Anda pada {day} pukul {time}. Balas C untuk mengonfirmasi atau hubungi klinik.",
        "known_invoice": "Faktur {invoice} dari {vendor} untuk pekerjaan yang selesai telah dilampirkan. Detail pembayaran tidak berubah.",
        "security_training": "Pelatihan keamanan {vendor} pada {day}: permintaan mendesak untuk kata sandi, kripto, atau kartu hadiah dapat merupakan penipuan. Laporkan pesan tersebut.",
        "executive_workflow": "Permintaan pembayaran {invoice} dari {vendor} berada dalam antrean keuangan yang disetujui. Gunakan detail pemasok yang tercatat; tidak ada transfer rahasia yang diminta.",
        "bank_notice": "{bank}: pembelian kartu sebesar {amount} disetujui pada {day}. Jika bukan Anda, hubungi nomor yang tertera pada kartu.",
        "recruiter_schedule": "{vendor} menjadwalkan wawancara Anda untuk posisi {role} pada {day}. Tidak ada pembayaran atau informasi bank yang diperlukan sebelum wawancara.",
        "delivery_pickup": "Paket {carrier} Anda siap diambil pada {day} pukul {time}. Bawa identitas; tidak ada pembayaran daring yang harus dilakukan.",
        "shared_document_safe": "{vendor} membagikan kontrak {invoice} di ruang kerja resmi. Buka aplikasi perusahaan yang sudah tersimpan; pesan ini tidak meminta kata sandi.",
        "official_app_alert": "{bank}: kami menolak transaksi sebesar {amount} pada {day}. Buka aplikasi resmi atau hubungi nomor pada kartu; kami tidak akan pernah meminta kode.",
        "parking_app_notice": "Sesi parkir {invoice} berakhir pada {day}. Jika perlu waktu tambahan, buka aplikasi parkir kota yang sudah terpasang; pesan ini tidak menyertakan tautan pembayaran.",
        "payroll_change_safe": "Pengingat penggajian {invoice} dari {vendor}: perubahan rekening sebelum {day} memerlukan pemeriksaan identitas langsung. Jangan kirim detail bank melalui surel atau obrolan.",
    },
    "Portuguese": {
        "otp_warning": "O seu código de verificação do {bank} é {digits}. Nunca partilhe este código; nenhum funcionário o pedirá.",
        "appointment": "Lembrete: a sua consulta de {clinic} é {day} às {time}. Responda C para confirmar ou ligue para o consultório.",
        "known_invoice": "A fatura {invoice} da {vendor} pelo trabalho concluído está anexada. Os dados de pagamento não mudaram.",
        "security_training": "Formação de segurança da {vendor} em {day}: pedidos urgentes de palavras-passe, criptomoedas ou cartões-presente podem ser fraude. Denuncie-os.",
        "executive_workflow": "O pedido de pagamento {invoice} da {vendor} está na fila financeira aprovada. Use os dados registados do fornecedor; não é pedida nenhuma transferência confidencial.",
        "bank_notice": "{bank}: uma compra com cartão de {amount} foi aprovada em {day}. Se não foi você, ligue para o número indicado no cartão.",
        "recruiter_schedule": "A {vendor} marcou a sua entrevista para a função de {role} em {day}. Não são necessários pagamentos nem dados bancários antes da entrevista.",
        "delivery_pickup": "A sua encomenda da {carrier} está pronta para levantamento em {day} às {time}. Leve identificação; não há qualquer pagamento online devido.",
        "shared_document_safe": "A {vendor} partilhou o contrato {invoice} no espaço de trabalho oficial. Abra a aplicação da empresa já guardada; esta mensagem não pede uma palavra-passe.",
        "official_app_alert": "{bank}: recusámos uma transação de {amount} em {day}. Abra a aplicação oficial ou ligue para o número no cartão; nunca pediremos um código.",
        "parking_app_notice": "A sessão de estacionamento {invoice} termina em {day}. Se precisar de mais tempo, abra a aplicação municipal já instalada; a mensagem não inclui ligação de pagamento.",
        "payroll_change_safe": "Lembrete de pagamento {invoice} da {vendor}: alterações à conta antes de {day} exigem verificação de identidade presencial. Não envie dados bancários por e-mail ou chat.",
    },
}

LANGUAGE_VALUES = {
    "Spanish": {
        "day": ["el lunes", "el jueves", "mañana", "el 28 de agosto"],
        "clinic": ["odontología", "oftalmología", "medicina familiar", "fisioterapia"],
        "role": ["asistente", "analista", "administrativo", "revisor"],
        "carrier": ["correos", "mensajería", "paquetería", "transporte"],
    },
    "Dutch": {
        "day": ["maandag", "donderdag", "morgen", "28 augustus"],
        "clinic": ["de tandarts", "de oogarts", "de huisarts", "fysiotherapie"],
        "role": ["assistent", "analist", "administratief medewerker", "beoordelaar"],
        "carrier": ["de post", "de koerier", "de pakketdienst", "de bezorgdienst"],
    },
    "French": {
        "day": ["lundi", "jeudi", "demain", "le 28 août"],
        "clinic": ["soins dentaires", "ophtalmologie", "médecine générale", "kinésithérapie"],
        "role": ["assistant", "analyste", "agent administratif", "réviseur"],
        "carrier": ["postal", "express", "de messagerie", "de livraison"],
    },
    "German": {
        "day": ["Montag", "Donnerstag", "morgen", "28. August"],
        "clinic": ["der Zahnärztin", "der Augenärztin", "der Hausarztpraxis", "der Physiotherapie"],
        "role": ["Assistenz", "Analyst", "Sachbearbeitung", "Prüfer"],
        "carrier": ["der Post", "dem Kurier", "dem Paketdienst", "dem Versanddienst"],
    },
    "Italian": {
        "day": ["lunedì", "giovedì", "domani", "28 agosto"],
        "clinic": ["dentistica", "oculistica", "medicina di famiglia", "fisioterapia"],
        "role": ["assistente", "analista", "impiegato amministrativo", "revisore"],
        "carrier": ["posta", "corriere", "spedizioni", "consegne"],
    },
    "Indonesian": {
        "day": ["Senin", "Kamis", "besok", "28 Agustus"],
        "clinic": ["dokter gigi", "dokter mata", "dokter keluarga", "fisioterapi"],
        "role": ["asisten", "analis", "staf administrasi", "peninjau"],
        "carrier": ["pos", "kurir", "layanan paket", "layanan pengiriman"],
    },
    "Portuguese": {
        "day": ["segunda-feira", "quinta-feira", "amanhã", "28 de agosto"],
        "clinic": ["medicina dentária", "oftalmologia", "medicina familiar", "fisioterapia"],
        "role": ["assistente", "analista", "administrativo", "revisor"],
        "carrier": ["correios", "transportadora", "serviço de encomendas", "serviço de entregas"],
    },
}

LANGUAGE_STYLES = {
    "Spanish": (
        "{text}",
        "Aviso: {text}",
        "ALERTA — {text}",
        "Mensaje automático: {text}",
        "Por favor, lee esto. {text}",
        "Recordatorio: {text}",
    ),
    "Dutch": (
        "{text}",
        "Bericht: {text}",
        "WAARSCHUWING — {text}",
        "Automatisch bericht: {text}",
        "Lees dit alstublieft. {text}",
        "Herinnering: {text}",
    ),
    "French": (
        "{text}",
        "Avis : {text}",
        "ALERTE — {text}",
        "Message automatique : {text}",
        "Veuillez lire. {text}",
        "Rappel : {text}",
    ),
    "German": (
        "{text}",
        "Hinweis: {text}",
        "WARNUNG — {text}",
        "Automatische Nachricht: {text}",
        "Bitte lesen. {text}",
        "Erinnerung: {text}",
    ),
    "Italian": (
        "{text}",
        "Avviso: {text}",
        "AVVISO — {text}",
        "Messaggio automatico: {text}",
        "Leggi con attenzione. {text}",
        "Promemoria: {text}",
    ),
    "Indonesian": (
        "{text}",
        "Pemberitahuan: {text}",
        "PERINGATAN — {text}",
        "Pesan otomatis: {text}",
        "Mohon dibaca. {text}",
        "Pengingat: {text}",
    ),
    "Portuguese": (
        "{text}",
        "Aviso: {text}",
        "ALERTA — {text}",
        "Mensagem automática: {text}",
        "Leia, por favor. {text}",
        "Lembrete: {text}",
    ),
}

STYLES = (
    "{text}",
    "Notice: {text}",
    "ALERT — {text}",
    "Automated message: {text}",
    "Please read. {text}",
    "Reminder: {text}",
)


FAMILY_SPLITS = {
    "delivery_toll": "train",
    "delivery_parcel": "dev",
    "bank_otp": "test",
    "bank_link": "train",
    "invoice_change": "train",
    "advance_fee": "train",
    "investment_guarantee": "train",
    "job_check": "dev",
    "job_fee": "train",
    "romance_emergency": "train",
    "authority_threat": "test",
    "tech_support": "train",
    "marketplace_offplatform": "test",
    "vendor_bank_change": "dev",
    "family_new_number": "test",
    "recruiter_identity": "dev",
    "refund_recovery": "train",
    "quiet_account_takeover": "test",
    "executive_payment": "train",
    "relationship_travel": "dev",
    "credential_shared_document": "dev",
    "identity_case_callback": "dev",
    "delivery_parking": "test",
    "financial_payroll_change": "test",
    "relationship_inheritance": "test",
    "benefit_identity_update": "train",
    "jury_duty_penalty": "train",
    "phantom_hacker_transfer": "train",
    "tax_refund_identity": "train",
    "task_unlock_deposit": "train",
    "bank_security_code": "train",
    "family_bail_intermediary": "train",
    "reshipping_job": "train",
    "medicare_card_fee": "train",
    "immigration_case_fee": "train",
    "package_identity_form": "train",
    "account_safe_harbor": "train",
    "otp_warning": "train",
    "bank_notice": "dev",
    "appointment": "train",
    "known_invoice": "train",
    "family_request": "test",
    "security_training": "train",
    "promotion": "train",
    "donation": "dev",
    "official_app_alert": "test",
    "recruiter_schedule": "dev",
    "marketplace_platform": "test",
    "family_transfer_verified": "test",
    "vendor_change_verified": "dev",
    "delivery_pickup": "dev",
    "executive_workflow": "train",
    "partner_travel_verified": "dev",
    "shared_document_safe": "dev",
    "police_callback_safe": "dev",
    "parking_app_notice": "test",
    "payroll_change_safe": "test",
    "relationship_known_contact": "test",
    "benefit_portal_safe": "train",
    "jury_duty_notice_safe": "train",
    "bank_transfer_warning_safe": "train",
    "tax_opt_in_safe": "train",
    "task_job_safe": "train",
    "bank_code_warning_safe": "train",
    "family_emergency_check_safe": "train",
    "reshipping_warning_safe": "train",
    "medicare_card_safe": "train",
    "immigration_case_safe": "train",
    "package_tracking_opt_in_safe": "train",
    "remote_support_booking_safe": "train",
}


def variants(
    template: str,
    count: int,
    rng: random.Random,
    values: dict[str, list[str]] | None = None,
    styles: tuple[str, ...] = STYLES,
) -> list[str]:
    active_values = values or VALUES
    names = [name for name in active_values if "{" + name + "}" in template]
    combinations = list(itertools.product(*(active_values[name] for name in names)))
    rng.shuffle(combinations)
    rendered = {
        style.format(text=template.format(**dict(zip(names, values, strict=True))))
        for values, style in itertools.product(combinations, styles)
    }
    rendered = sorted(rendered)
    rng.shuffle(rendered)
    if len(rendered) < count:
        raise ValueError(f"template has only {len(rendered)} unique variants; requested {count}")
    return rendered[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/generated/synthetic.jsonl"))
    parser.add_argument("--per-family", type=int, default=72)
    parser.add_argument("--multilingual-per-family", type=int, default=36)
    parser.add_argument("--targeted-per-family", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if args.per_family < 3:
        raise ValueError("--per-family must be at least 3")
    if args.multilingual_per_family < 3:
        raise ValueError("--multilingual-per-family must be at least 3")
    if args.targeted_per_family < 3:
        raise ValueError("--targeted-per-family must be at least 3")

    rng = random.Random(args.seed)
    rows = []
    families = [("SCAM", SCAM_FAMILIES), (None, HARD_NEGATIVE_FAMILIES)]
    for default_label, definitions in families:
        for family, (category_or_label, template) in definitions.items():
            label = default_label or category_or_label
            category = category_or_label if default_label else "NONE"
            family_id = f"synthetic:{family}:v{GENERATOR_VERSION}"
            for text in variants(template, args.per_family, rng):
                digest = hashlib.sha256(text.encode()).hexdigest()[:16]
                rows.append(
                    {
                        "id": f"syn-{digest}",
                        "text": text,
                        "label": label,
                        "category": category,
                        "source": f"scamguard_synthetic_v{GENERATOR_VERSION}",
                        "source_label": label.lower(),
                        "license": "Apache-2.0",
                        "split": FAMILY_SPLITS[family],
                        "family_id": family_id,
                        "is_synthetic": True,
                        "synthetic_method": "deterministic_slot_filling_original_copy",
                        "pattern_reference": FAMILY_REFERENCE_URLS.get(
                            family, SYNTHETIC_REFERENCE_DEFAULT
                        ),
                    }
                )

    for language, definitions in MULTILINGUAL_SAFE_TEMPLATES.items():
        language_values = VALUES | LANGUAGE_VALUES[language]
        for family, template in definitions.items():
            family_id = f"synthetic:multilingual:{family}:{language.casefold()}:v{GENERATOR_VERSION}"
            for text in variants(
                template,
                args.multilingual_per_family,
                rng,
                language_values,
                LANGUAGE_STYLES[language],
            ):
                digest = hashlib.sha256(text.encode()).hexdigest()[:16]
                rows.append(
                    {
                        "id": f"syn-{digest}",
                        "text": text,
                        "label": "SAFE",
                        "category": "NONE",
                        "source": f"scamguard_synthetic_v{GENERATOR_VERSION}",
                        "source_label": "safe_multilingual_hard_negative",
                        "source_language": language,
                        "license": "Apache-2.0",
                        "split": FAMILY_SPLITS[family],
                        "family_id": family_id,
                        "is_synthetic": True,
                        "synthetic_method": "deterministic_slot_filling_original_copy",
                        "pattern_reference": SYNTHETIC_REFERENCE_DEFAULT,
                    }
                )

    targeted_rng = random.Random(f"{args.seed}:targeted-counterfactual-v1")
    for family, (label, category, template, reference) in TARGETED_COUNTERFACTUAL_FAMILIES.items():
        family_id = f"synthetic:counterfactual:{family}:v{TARGETED_COUNTERFACTUAL_VERSION}"
        for text in variants(template, args.targeted_per_family, targeted_rng):
            digest = hashlib.sha256(text.encode()).hexdigest()[:16]
            rows.append(
                {
                    "id": f"syn-{digest}",
                    "text": text,
                    "label": label,
                    "category": category,
                    "source": (
                        "scamguard_synthetic_counterfactual_"
                        f"v{TARGETED_COUNTERFACTUAL_VERSION}"
                    ),
                    "source_label": label.lower(),
                    "license": "Apache-2.0",
                    "split": "train",
                    "family_id": family_id,
                    "is_synthetic": True,
                    "synthetic_method": (
                        "paired_deterministic_slot_filling_error_audit_grounded_original_copy"
                    ),
                    "pattern_reference": reference,
                    "generator_version": TARGETED_COUNTERFACTUAL_VERSION,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output.parent / "synthetic_manifest.json").write_text(
        json.dumps(
            {
                "generator_version": GENERATOR_VERSION,
                "seed": args.seed,
                "per_family": args.per_family,
                "multilingual_per_family": args.multilingual_per_family,
                "targeted_counterfactual_version": TARGETED_COUNTERFACTUAL_VERSION,
                "targeted_per_family": args.targeted_per_family,
                "targeted_counterfactual_families": sorted(TARGETED_COUNTERFACTUAL_FAMILIES),
                "scam_families": sorted(SCAM_FAMILIES),
                "hard_negative_families": sorted(HARD_NEGATIVE_FAMILIES),
                "multilingual_safe_languages": sorted(MULTILINGUAL_SAFE_TEMPLATES),
                "multilingual_safe_families": sorted(
                    next(iter(MULTILINGUAL_SAFE_TEMPLATES.values()))
                ),
                "method": "deterministic slot filling with original copy",
                "pattern_references": sorted(
                    {SYNTHETIC_REFERENCE_DEFAULT, *FAMILY_REFERENCE_URLS.values()}
                ),
                "rows": len(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(rows)} synthetic examples across base and multilingual hard-negative families"
    )


if __name__ == "__main__":
    main()
