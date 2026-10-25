"""Knowledge base ingestion pipeline and seed corpus for customer support domains."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from src.core.config import settings
from src.rag.chunking import (
    chunk_article,
    deduplicate_and_enrich,
    normalize_category,
)
from src.rag.store import ChromaVectorStore

logger = logging.getLogger(__name__)

# Built-in seed corpus (70+ rich, realistic rows across all 6 core categories + excluded categories)
DEFAULT_BITEXT_SEED_ROWS: list[dict[str, str]] = [
    # ==========================================
    # BILLING (PAYMENT / INVOICE)
    # ==========================================
    {
        "category": "INVOICE",
        "intent": "check_invoice",
        "instruction": "Where can I download my latest invoice?",
        "response": "You can view and download all past invoices from the Billing section of your Account Dashboard. Invoices are generated as PDF files within 24 hours of successful payment.",
    },
    {
        "category": "INVOICE",
        "intent": "check_invoice",
        "instruction": "I need a copy of my receipt for my tax records.",
        "response": "You can view and download all past invoices from the Billing section of your Account Dashboard. Invoices are generated as PDF files within 24 hours of successful payment.",
    },
    {
        "category": "BILLING",
        "intent": "check_invoice",
        "instruction": "Can you email me my monthly billing statement?",
        "response": "You can view and download all past invoices from the Billing section of your Account Dashboard. Invoices are generated as PDF files within 24 hours of successful payment.",
    },
    {
        "category": "PAYMENT",
        "intent": "payment_issue",
        "instruction": "My credit card was declined at checkout.",
        "response": "Payment failures typically occur due to insufficient funds, incorrect CVV or billing zip codes, or automated bank security blocks. Please verify your billing address or try an alternative payment method.",
    },
    {
        "category": "PAYMENT",
        "intent": "payment_issue",
        "instruction": "Why does my transaction say payment authorization failed?",
        "response": "Payment failures typically occur due to insufficient funds, incorrect CVV or billing zip codes, or automated bank security blocks. Please verify your billing address or try an alternative payment method.",
    },
    {
        "category": "PAYMENT",
        "intent": "payment_issue",
        "instruction": "Checkout keeps rejecting my card.",
        "response": "Payment failures typically occur due to insufficient funds, incorrect CVV or billing zip codes, or automated bank security blocks. Please verify your billing address or try an alternative payment method.",
    },
    {
        "category": "BILLING",
        "intent": "dispute_charge",
        "instruction": "I see an unfamiliar charge on my bank statement from your store.",
        "response": "If you notice an unrecognized charge, please verify if an authorized family member made a purchase. To dispute an erroneous charge, provide the date, last 4 digits of the card, and amount to billing support for immediate review.",
    },
    {
        "category": "PAYMENT",
        "intent": "dispute_charge",
        "instruction": "I was charged twice for the same order.",
        "response": "If you notice an unrecognized charge, please verify if an authorized family member made a purchase. To dispute an erroneous charge, provide the date, last 4 digits of the card, and amount to billing support for immediate review.",
    },
    {
        "category": "INVOICE",
        "intent": "apply_promotional_discount",
        "instruction": "My coupon code did not apply at checkout.",
        "response": "Promotional codes must be entered in the promo code field before finalizing order payment. If a valid promotional code failed to apply due to a technical error, contact billing support with your order number for a retroactive credit adjustment.",
    },
    {
        "category": "BILLING",
        "intent": "apply_promotional_discount",
        "instruction": "Can you apply a discount promo code after I already paid?",
        "response": "Promotional codes must be entered in the promo code field before finalizing order payment. If a valid promotional code failed to apply due to a technical error, contact billing support with your order number for a retroactive credit adjustment.",
    },
    {
        "category": "PAYMENT",
        "intent": "apply_promotional_discount",
        "instruction": "The 20 percent off coupon was not deducted from my invoice.",
        "response": "Promotional codes must be entered in the promo code field before finalizing order payment. If a valid promotional code failed to apply due to a technical error, contact billing support with your order number for a retroactive credit adjustment.",
    },
    # ==========================================
    # SHIPPING (DELIVERY / SHIPPING_ADDRESS)
    # ==========================================
    {
        "category": "DELIVERY",
        "intent": "delivery_options",
        "instruction": "What delivery methods and speeds are available?",
        "response": "We offer Standard Shipping (3-5 business days), Expedited Shipping (2 business days), and Next-Day Express Delivery. Shipping costs are calculated at checkout based on weight and destination.",
    },
    {
        "category": "SHIPPING",
        "intent": "delivery_options",
        "instruction": "Do you offer overnight delivery?",
        "response": "We offer Standard Shipping (3-5 business days), Expedited Shipping (2 business days), and Next-Day Express Delivery. Shipping costs are calculated at checkout based on weight and destination.",
    },
    {
        "category": "DELIVERY",
        "intent": "delivery_options",
        "instruction": "How long does standard delivery take to arrive?",
        "response": "We offer Standard Shipping (3-5 business days), Expedited Shipping (2 business days), and Next-Day Express Delivery. Shipping costs are calculated at checkout based on weight and destination.",
    },
    {
        "category": "SHIPPING_ADDRESS",
        "intent": "change_shipping_address",
        "instruction": "I entered the wrong delivery address on my recent order.",
        "response": "Shipping addresses can be modified within 60 minutes of order placement if the package has not yet entered fulfillment. Contact support immediately with your order number and updated address.",
    },
    {
        "category": "SHIPPING_ADDRESS",
        "intent": "change_shipping_address",
        "instruction": "Can I redirect my package to a new apartment address?",
        "response": "Shipping addresses can be modified within 60 minutes of order placement if the package has not yet entered fulfillment. Contact support immediately with your order number and updated address.",
    },
    {
        "category": "SHIPPING",
        "intent": "change_shipping_address",
        "instruction": "Update my delivery address before shipment.",
        "response": "Shipping addresses can be modified within 60 minutes of order placement if the package has not yet entered fulfillment. Contact support immediately with your order number and updated address.",
    },
    {
        "category": "DELIVERY",
        "intent": "track_shipment",
        "instruction": "How can I track the carrier delivering my package?",
        "response": "Once your order dispatches, you will receive an email containing a carrier tracking link (FedEx, UPS, or USPS). You can also monitor live carrier tracking in the Orders tab of your account.",
    },
    {
        "category": "SHIPPING",
        "intent": "track_shipment",
        "instruction": "Tracking says my package is delayed in transit.",
        "response": "Once your order dispatches, you will receive an email containing a carrier tracking link (FedEx, UPS, or USPS). You can also monitor live carrier tracking in the Orders tab of your account.",
    },
    {
        "category": "DELIVERY",
        "intent": "switch_shipping_method",
        "instruction": "Can I upgrade my order to express shipping after ordering?",
        "response": "Shipping speeds may be upgraded within 45 minutes of purchase prior to order picking. Contact customer service with your order details to pay the difference for expedited or overnight delivery.",
    },
    {
        "category": "SHIPPING",
        "intent": "switch_shipping_method",
        "instruction": "Change standard shipping to next day delivery.",
        "response": "Shipping speeds may be upgraded within 45 minutes of purchase prior to order picking. Contact customer service with your order details to pay the difference for expedited or overnight delivery.",
    },
    # ==========================================
    # CANCELLATION (CANCELLATION_FEE / ORDER cancel_order)
    # ==========================================
    {
        "category": "ORDER",
        "intent": "cancel_order",
        "instruction": "I want to cancel my order immediately.",
        "response": "Orders may be cancelled free of charge within 30 minutes of placement before warehouse dispatch. Once an order is processed for packing, it cannot be cancelled and must be returned after delivery.",
    },
    {
        "category": "ORDER",
        "intent": "cancel_order",
        "instruction": "Please stop shipment and cancel order #98234.",
        "response": "Orders may be cancelled free of charge within 30 minutes of placement before warehouse dispatch. Once an order is processed for packing, it cannot be cancelled and must be returned after delivery.",
    },
    {
        "category": "ORDER",
        "intent": "cancel_order",
        "instruction": "I changed my mind, how do I cancel my order?",
        "response": "Orders may be cancelled free of charge within 30 minutes of placement before warehouse dispatch. Once an order is processed for packing, it cannot be cancelled and must be returned after delivery.",
    },
    {
        "category": "CANCELLATION_FEE",
        "intent": "check_cancellation_fee",
        "instruction": "Is there a penalty or fee for cancelling my order?",
        "response": "There are no cancellation fees if the order is cancelled within the 30-minute grace period. Custom-manufactured items and expedited freight shipments may be subject to a 15% restocking fee if cancelled after processing starts.",
    },
    {
        "category": "CANCELLATION_FEE",
        "intent": "check_cancellation_fee",
        "instruction": "Do you charge a restocking fee if I cancel after it ships?",
        "response": "There are no cancellation fees if the order is cancelled within the 30-minute grace period. Custom-manufactured items and expedited freight shipments may be subject to a 15% restocking fee if cancelled after processing starts.",
    },
    {
        "category": "CANCELLATION",
        "intent": "check_cancellation_fee",
        "instruction": "Will I get all my money back if I cancel?",
        "response": "There are no cancellation fees if the order is cancelled within the 30-minute grace period. Custom-manufactured items and expedited freight shipments may be subject to a 15% restocking fee if cancelled after processing starts.",
    },
    # ==========================================
    # ORDER (Non-cancellation order intents)
    # ==========================================
    {
        "category": "ORDER",
        "intent": "place_order",
        "instruction": "How do I place an order online?",
        "response": "To place an order, add your desired items to the shopping cart, proceed to checkout, provide your shipping details, and select your preferred payment method. You will receive an email confirmation upon completion.",
    },
    {
        "category": "ORDER",
        "intent": "place_order",
        "instruction": "Can I place an order over the phone with an agent?",
        "response": "To place an order, add your desired items to the shopping cart, proceed to checkout, provide your shipping details, and select your preferred payment method. You will receive an email confirmation upon completion.",
    },
    {
        "category": "ORDER",
        "intent": "place_order",
        "instruction": "Can I place a bulk order for my company?",
        "response": "To place an order, add your desired items to the shopping cart, proceed to checkout, provide your shipping details, and select your preferred payment method. You will receive an email confirmation upon completion.",
    },
    {
        "category": "ORDER",
        "intent": "change_order",
        "instruction": "Can I add another item to an order I just placed?",
        "response": "We cannot modify items or quantities in an existing order once submitted. You may place a new order for additional items, or cancel the current order within 30 minutes and reorder.",
    },
    {
        "category": "ORDER",
        "intent": "change_order",
        "instruction": "I selected the wrong size, how do I change the order?",
        "response": "We cannot modify items or quantities in an existing order once submitted. You may place a new order for additional items, or cancel the current order within 30 minutes and reorder.",
    },
    {
        "category": "ORDER",
        "intent": "change_order",
        "instruction": "How do I switch the color of an item in my order?",
        "response": "We cannot modify items or quantities in an existing order once submitted. You may place a new order for additional items, or cancel the current order within 30 minutes and reorder.",
    },
    {
        "category": "ORDER",
        "intent": "track_order",
        "instruction": "Where is my order? Can you check the status?",
        "response": "You can check your order status anytime by entering your order number and email on our Order Lookup page. Status updates include Order Placed, Processing, Dispatched, and Delivered.",
    },
    {
        "category": "ORDER",
        "intent": "track_order",
        "instruction": "What is the status of order #45129?",
        "response": "You can check your order status anytime by entering your order number and email on our Order Lookup page. Status updates include Order Placed, Processing, Dispatched, and Delivered.",
    },
    {
        "category": "ORDER",
        "intent": "track_order",
        "instruction": "Has my package left the warehouse yet?",
        "response": "You can check your order status anytime by entering your order number and email on our Order Lookup page. Status updates include Order Placed, Processing, Dispatched, and Delivered.",
    },
    # ==========================================
    # ACCOUNT
    # ==========================================
    {
        "category": "ACCOUNT",
        "intent": "create_account",
        "instruction": "How do I sign up for a new customer account?",
        "response": "To register an account, click 'Sign Up' at the top right of the homepage, enter your email and password, and verify your email address via the confirmation link sent to your inbox.",
    },
    {
        "category": "ACCOUNT",
        "intent": "create_account",
        "instruction": "Do I need an account to make a purchase or can I checkout as guest?",
        "response": "To register an account, click 'Sign Up' at the top right of the homepage, enter your email and password, and verify your email address via the confirmation link sent to your inbox.",
    },
    {
        "category": "ACCOUNT",
        "intent": "create_account",
        "instruction": "Creating an account for rewards points.",
        "response": "To register an account, click 'Sign Up' at the top right of the homepage, enter your email and password, and verify your email address via the confirmation link sent to your inbox.",
    },
    {
        "category": "ACCOUNT",
        "intent": "recover_password",
        "instruction": "I forgot my password and cannot sign in.",
        "response": "Click 'Forgot Password' on the login screen and enter your registered email. A secure password reset link will be sent to your email, valid for 60 minutes.",
    },
    {
        "category": "ACCOUNT",
        "intent": "recover_password",
        "instruction": "My account is locked after too many failed password attempts.",
        "response": "Click 'Forgot Password' on the login screen and enter your registered email. A secure password reset link will be sent to your email, valid for 60 minutes.",
    },
    {
        "category": "ACCOUNT",
        "intent": "recover_password",
        "instruction": "How do I reset my account login credentials?",
        "response": "Click 'Forgot Password' on the login screen and enter your registered email. A secure password reset link will be sent to your email, valid for 60 minutes.",
    },
    {
        "category": "ACCOUNT",
        "intent": "edit_account",
        "instruction": "How do I change the email address linked to my account?",
        "response": "To update profile information, sign in and navigate to 'Account Settings'. You can edit your name, default shipping addresses, telephone number, and communication preferences.",
    },
    {
        "category": "ACCOUNT",
        "intent": "edit_account",
        "instruction": "Where do I update my phone number on my profile?",
        "response": "To update profile information, sign in and navigate to 'Account Settings'. You can edit your name, default shipping addresses, telephone number, and communication preferences.",
    },
    {
        "category": "ACCOUNT",
        "intent": "edit_account",
        "instruction": "Delete or update my saved payment cards in my account.",
        "response": "To update profile information, sign in and navigate to 'Account Settings'. You can edit your name, default shipping addresses, telephone number, and communication preferences.",
    },
    {
        "category": "ACCOUNT",
        "intent": "delete_account",
        "instruction": "How do I permanently delete my account and personal data?",
        "response": "Under privacy regulations (GDPR/CCPA), you can request account deletion by submitting a data erasure request in Account Privacy Settings. Your account, profile data, and saved payment methods will be permanently erased within 30 days.",
    },
    {
        "category": "ACCOUNT",
        "intent": "delete_account",
        "instruction": "Close my account and purge my personal information.",
        "response": "Under privacy regulations (GDPR/CCPA), you can request account deletion by submitting a data erasure request in Account Privacy Settings. Your account, profile data, and saved payment methods will be permanently erased within 30 days.",
    },
    # ==========================================
    # REFUND
    # ==========================================
    {
        "category": "REFUND",
        "intent": "get_refund",
        "instruction": "How do I request a refund for an item I purchased?",
        "response": "Full refunds are available within 30 days of purchase for unused items in original packaging. To initiate a refund, submit a return request through the Returns Portal or contact customer service.",
    },
    {
        "category": "REFUND",
        "intent": "get_refund",
        "instruction": "What is your refund policy window?",
        "response": "Full refunds are available within 30 days of purchase for unused items in original packaging. To initiate a refund, submit a return request through the Returns Portal or contact customer service.",
    },
    {
        "category": "REFUND",
        "intent": "get_refund",
        "instruction": "I want my money back for an unwanted item.",
        "response": "Full refunds are available within 30 days of purchase for unused items in original packaging. To initiate a refund, submit a return request through the Returns Portal or contact customer service.",
    },
    {
        "category": "REFUND",
        "intent": "track_refund",
        "instruction": "When will my refund show up in my bank account?",
        "response": "Once our warehouse inspects your returned merchandise, approved refunds are processed within 3-5 business days. Depending on your financial institution, funds appear on your statement in 5-10 business days.",
    },
    {
        "category": "REFUND",
        "intent": "track_refund",
        "instruction": "I have not received my credit refund yet.",
        "response": "Once our warehouse inspects your returned merchandise, approved refunds are processed within 3-5 business days. Depending on your financial institution, funds appear on your statement in 5-10 business days.",
    },
    {
        "category": "REFUND",
        "intent": "track_refund",
        "instruction": "Check the status of my return refund payout.",
        "response": "Once our warehouse inspects your returned merchandise, approved refunds are processed within 3-5 business days. Depending on your financial institution, funds appear on your statement in 5-10 business days.",
    },
    {
        "category": "REFUND",
        "intent": "refund_damaged_item",
        "instruction": "My package arrived broken and damaged. How do I get a refund?",
        "response": "For items that arrive damaged or defective, we provide an immediate free replacement or 100% refund without requiring return shipping. Please provide your order number and photos of the damaged item.",
    },
    {
        "category": "REFUND",
        "intent": "refund_damaged_item",
        "instruction": "The item inside the box was cracked during delivery.",
        "response": "For items that arrive damaged or defective, we provide an immediate free replacement or 100% refund without requiring return shipping. Please provide your order number and photos of the damaged item.",
    },
    {
        "category": "REFUND",
        "intent": "refund_damaged_item",
        "instruction": "Received a defective product, I need an immediate reimbursement.",
        "response": "For items that arrive damaged or defective, we provide an immediate free replacement or 100% refund without requiring return shipping. Please provide your order number and photos of the damaged item.",
    },
    {
        "category": "REFUND",
        "intent": "refund_damaged_item",
        "instruction": "What photos or evidence do I need to submit to get a refund for a broken item?",
        "response": "For items that arrive damaged or defective, we provide an immediate free replacement or 100% refund without requiring return shipping. Please provide your order number and photos of the damaged item.",
    },
    {
        "category": "REFUND",
        "intent": "refund_damaged_item",
        "instruction": "Can I get a refund if the delivery box arrived crushed and the goods inside are ruined?",
        "response": "For items that arrive damaged or defective, we provide an immediate free replacement or 100% refund without requiring return shipping. Please provide your order number and photos of the damaged item.",
    },
    # ==========================================
    # EXCLUDED DOMAINS (FEEDBACK / NEWSLETTER) -> Must be filtered out
    # ==========================================
    {
        "category": "FEEDBACK",
        "intent": "give_feedback",
        "instruction": "I wanted to let you know how great your service was yesterday.",
        "response": "Thank you for taking the time to share your feedback with us.",
    },
    {
        "category": "FEEDBACK",
        "intent": "leave_review",
        "instruction": "Where can I leave a product review for the shoes I bought?",
        "response": "You can leave product reviews directly on the product catalog page.",
    },
    {
        "category": "FEEDBACK",
        "intent": "complaint",
        "instruction": "I was unhappy with the wait time on customer support.",
        "response": "We apologize for the delay and appreciate your constructive feedback.",
    },
    {
        "category": "NEWSLETTER",
        "intent": "subscribe_newsletter",
        "instruction": "How do I subscribe to promotional discount emails?",
        "response": "Enter your email address in the newsletter signup box at the footer of our website.",
    },
    {
        "category": "NEWSLETTER",
        "intent": "unsubscribe_newsletter",
        "instruction": "Stop sending me marketing newsletters.",
        "response": "Click the unsubscribe link at the bottom of any marketing email.",
    },
]


def load_bitext_dataset(
    use_hf: bool = False,
    max_rows: int = 5000,
) -> list[dict[str, Any]]:
    """Load customer support rows from HuggingFace dataset or local seed corpus."""
    if use_hf:
        try:
            logger.info("Attempting to load Bitext dataset from HuggingFace...")
            from datasets import load_dataset

            hf_ds = load_dataset(
                "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                split="train",
            )
            rows: list[dict[str, Any]] = []
            for item in hf_ds:
                rows.append(
                    {
                        "category": item.get("category", ""),
                        "intent": item.get("intent", ""),
                        "instruction": item.get("instruction", ""),
                        "response": item.get("response", ""),
                    }
                )
                if len(rows) >= max_rows:
                    break
            logger.info(f"Loaded {len(rows)} rows from HuggingFace dataset.")
            return rows
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Failed to load dataset from HuggingFace ({e}). Falling back to seed corpus."
            )

    return list(DEFAULT_BITEXT_SEED_ROWS)


def run_ingestion(
    persist_dir: str | None = None,
    output_json_path: str = "data/knowledge_base/articles.json",
    use_hf: bool = False,
) -> dict[str, Any]:
    """Execute complete ingestion pipeline: load, normalize, deduplicate, chunk, and index."""
    start_time = time.perf_counter()

    # 1. Load dataset rows
    raw_rows = load_bitext_dataset(use_hf=use_hf)

    # 2. Filter check
    filtered_rows = [
        r
        for r in raw_rows
        if normalize_category(
            r.get("category") or r.get("raw_category"),
            r.get("intent"),
        )
        is not None
    ]

    # 3. Deduplicate and enrich
    articles = deduplicate_and_enrich(raw_rows)

    # 4. Chunk articles
    chunks: list[dict[str, Any]] = []
    for art in articles:
        chunks.extend(chunk_article(art))

    # 5. Calculate token usage via tiktoken
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = sum(len(enc.encode(c.get("content", ""))) for c in chunks)
    except Exception:  # noqa: BLE001
        total_tokens = sum(len(c.get("content", "").split()) for c in chunks)

    # 6. Initialize Chroma store and index documents
    target_persist_dir = persist_dir or settings.chroma_persist_dir
    vector_store = ChromaVectorStore(persist_dir=target_persist_dir)
    vector_store.reset()
    vector_store.add_documents(chunks)

    # 7. Persist articles JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    stats = {
        "raw_rows": len(raw_rows),
        "filtered_rows": len(filtered_rows),
        "articles_created": len(articles),
        "chunks_indexed": len(chunks),
        "categories": sorted({a["category"] for a in articles}),
        "elapsed_ms": round(elapsed_ms, 2),
        "total_embedding_tokens": total_tokens,
    }

    logger.info(
        f"Ingestion complete: {stats['articles_created']} articles ({stats['chunks_indexed']} chunks) "
        f"indexed across {len(stats['categories'])} categories in {stats['elapsed_ms']}ms."
    )
    return stats
